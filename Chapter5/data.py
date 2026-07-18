import typing as tt
import kagglehub as kh
import pathlib
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
import albumentations as A


CLASSES_TRANSLATE = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}

# The side of the image after rescale
TARGET_SIZE = 300
# Color to pad the image
FILL_COLOR = 128
# shuffle seed
TRAIN_TEST_DEFAULT_SEED = 112

# data sample type - pair of image path and label
TSample = tt.Tuple[pathlib.Path, str]

# Normalization parameters computed in 01-eda.ipynb on train+val
# datasets with TRAIN_TEST_DEFAULT_SEED
NORM_MEAN = (0.51822103, 0.50493532, 0.44138216)
NORM_STD  = (0.23438585, 0.23087398, 0.24867558)


# Transformations applied before the optional augmentation
RESCALE_TRANSFORM = A.Compose([
    A.LongestMaxSize(TARGET_SIZE),
    A.PadIfNeeded(TARGET_SIZE, TARGET_SIZE, fill=FILL_COLOR),
])

NORM_TRANSFORM = A.Normalize(
    mean=NORM_MEAN,
    std=NORM_STD,
    max_pixel_value=255,
)

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


def get_english_classes(class_names: tt.List[str]) -> tt.List[str]:
    return [
        CLASSES_TRANSLATE[name]
        for name in class_names
    ]


def get_class_files(class_path: pathlib.Path) -> tt.List[pathlib.Path]:
    res = list(class_path.glob("*.jpeg"))
    res.sort()
    return res


def get_class_weights(
        path: pathlib.Path, class_names: tt.List[str]
) -> tt.List[float]:
    """
    Get relative weights of classes (classes with less samples have more weight)
    :param path: top data path
    :param class_names: names of classes in the same order as they will be used during the training.
    :return: list of class weights based on samples count
    """
    counts = []
    for class_name in class_names:
        class_path = path / class_name
        counts.append(len(list(class_path.glob("*.jpeg"))))
    max_cnt = max(counts)
    return [
        max_cnt / count
        for count in counts
    ]


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
    res *= np.array(NORM_STD) * 255
    res += np.array(NORM_MEAN) * 255
    res = res.astype(np.uint8)
    return res


class ImagesDataset(Dataset):
    def __init__(self, classes: tt.List[str], samples: tt.List[TSample],
                 rescale: tt.Optional[A.Compose] = None,
                 augment: tt.Optional[A.Compose] = None):
        self.classes = classes
        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(classes)
        }
        self.samples = samples
        self.rescale = rescale if rescale is not None else RESCALE_TRANSFORM
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item):
        file_path, class_name = self.samples[item]
        class_idx = self.class_to_idx[class_name]
        img = cv2.imread(file_path, flags=cv2.IMREAD_COLOR_RGB)
        scaled_img = self.rescale(image=img)['image']
        if self.augment is not None:
            scaled_img = self.augment(image=scaled_img)['image']
        norm_img = NORM_TRANSFORM(image=scaled_img)['image']
        res_img = norm_img.transpose((2, 0, 1))
        return res_img, class_idx
