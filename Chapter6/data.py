import typing as tt
import kagglehub as kh
import pathlib
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
import albumentations as A


# The side of the image after cropping
TARGET_SIZE = 128
# Color to pad the image
FILL_COLOR = 128
# shuffle seed
TRAIN_TEST_DEFAULT_SEED = 112

# data sample type - pair of image path and label
TSample = tt.Tuple[pathlib.Path, str]

# Transformations applied before the noise
PREPROCESS_TRANSFORM = A.Compose([
    A.RandomCrop(TARGET_SIZE, TARGET_SIZE, fill=FILL_COLOR, pad_if_needed=True),
    A.Normalize(normalization="min_max", max_pixel_value=255),
])


def get_images_path() -> pathlib.Path:
    path = kh.dataset_download("alessiocorrado99/animals10")
    return pathlib.Path(path) / "raw-img"


def get_classes(path: pathlib.Path) -> tt.List[str]:
    """
    Get sorted list of classes
    :param path: images path
    :return: list of sorted classes
    """
    res = []
    for p in path.glob("*"):
        if not p.is_dir():
            continue
        res.append(p.name)
    res.sort()
    return res


def get_class_files(class_path: pathlib.Path) -> tt.List[pathlib.Path]:
    res = list(class_path.glob("*.jpeg"))
    res.sort()
    return res


def get_sorted_samples(path: pathlib.Path) -> tt.List[TSample]:
    """
    Get list of pairs (file path, label) in a sorted order.
    :param path: base images path
    :return: sorted samples
    """
    res = []
    for class_name in get_classes(path):
        for file_name in get_class_files(path / class_name):
            res.append((file_name, class_name))
    res.sort()
    return res


def split_train_val_test(samples: tt.List[tt.Any],
                         seed: tt.Optional[int] = None, shuffle: bool = True) \
        -> tt.Tuple[tt.List[tt.Any], tt.List[tt.Any], tt.List[tt.Any]]:
    """
    Split dataset into three parts 80%, 10% and 10%
    :param samples: list of samples to split
    :param seed: random seed to use (optional)
    :param shuffle: perform shuffle of the list before split
    :return: split parts
    """
    if seed is not None:
        random.seed(seed)
    if shuffle:
        random.shuffle(samples)
    l1 = int(round(len(samples) * .8))
    l2 = (len(samples) - l1) >> 1
    return samples[:l1], samples[l1:l1+l2], samples[l1+l2:]


def denorm_image(img: np.ndarray) -> np.ndarray:
    """
    Denormalize image - return back into uint8 0-255 range
    """
    res = np.array(img, copy=True)
    res = res.transpose((1, 2, 0))
    res *= 255
    res = res.astype(np.uint8)
    return res


class DenoiseImagesDataset(Dataset):
    def __init__(self, samples: tt.List[TSample], noise: A.Compose):
        self.samples = samples
        self.noise = noise

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item):
        file_path, _ = self.samples[item]
        img = cv2.imread(file_path, flags=cv2.IMREAD_COLOR_RGB)
        prep_img = PREPROCESS_TRANSFORM(image=img)['image']
        noisy_img = self.noise(image=prep_img)['image']
        src_img = noisy_img.transpose((2, 0, 1))
        tgt_img = prep_img.transpose((2, 0, 1))
        return src_img, tgt_img
