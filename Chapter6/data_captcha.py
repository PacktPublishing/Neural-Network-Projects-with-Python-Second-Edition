# Functions related to captcha dataset
import kagglehub as kh
import cv2
import random
import typing as tt
import numpy as np
import pathlib
import string
import itertools
from PIL import Image, ImageFont, ImageDraw
from torch.utils.data import Dataset
import albumentations as A


# dimensions of single character cell
CELL_WIDTH = 30
CELL_HEIGHT = 40
TARGET_FONT = ImageFont.truetype("Arial.ttf", CELL_HEIGHT)
COLOR_BACK = "white"
COLOR_TEXT = "black"
TRAIN_TEST_DEFAULT_SEED = 124

TSample = tt.Tuple[pathlib.Path, str]

CAPTCHA_CHARS = tuple(
    str(c)
    for c in itertools.chain(string.ascii_letters, string.digits)
)


def get_images_path() -> pathlib.Path:
    path = kh.dataset_download("parsasam/captcha-dataset")
    return pathlib.Path(path)


def get_samples(images_path: pathlib.Path) -> tt.List[TSample]:
    res = [
        (p, p.stem)
        for p in images_path.glob("*.jpg")
    ]
    res.sort()
    return res


def split_train_val_test(samples: tt.List[TSample],
                         seed: tt.Optional[int] = None, shuffle: bool = True) \
        -> tt.Tuple[tt.List[TSample], tt.List[TSample], tt.List[TSample]]:
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


def make_target_image(captcha: str) -> Image:
    """
    Make grayscale target image from given captcha text
    :param captcha: text to render
    :return: PIL Image object
    """
    chars = len(captcha)
    text_img = Image.new("L", (chars * CELL_WIDTH, CELL_HEIGHT), COLOR_BACK)
    draw = ImageDraw.Draw(text_img)
    ofs_x = CELL_WIDTH / 2
    ofs_y = CELL_HEIGHT / 2
    for (i, c) in enumerate(captcha):
        draw.text(
            (ofs_x + CELL_WIDTH * i, ofs_y), c,
            fill=COLOR_TEXT, anchor="mm", font=TARGET_FONT)
    return text_img


class OCRDataset(Dataset):
    def __init__(self, size_mul: int = 1):
        self.size_mul = size_mul
        self.cache = dict()
        self.postprocess = A.Compose([
            A.Normalize(normalization="min_max", max_pixel_value=255),
            A.GaussianBlur(p=0.5),
            A.ToTensorV2(),
        ])

    def __len__(self):
        return len(CAPTCHA_CHARS) * self.size_mul

    def __getitem__(self, item):
        item %= len(CAPTCHA_CHARS)
        res = self.cache.get(item)
        if res is not None:
            return res, item
        img = make_target_image(CAPTCHA_CHARS[item])
        img_np = np.array(img)
        res = self.postprocess(image=img_np)['image']
        self.cache[item] = res
        return res, item



class CaptchaDataset(Dataset):
    def __init__(self, samples: tt.List[TSample]):
        self.samples = samples
        self.postprocess = A.Compose([
            A.Normalize(normalization="min_max", max_pixel_value=255),
            A.ToTensorV2(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        file_path, label = self.samples[item]
        img = cv2.imread(file_path, flags=cv2.IMREAD_COLOR_RGB)
        norm_img = self.postprocess(image=img)['image']
        tgt_img = np.array(make_target_image(label))
        norm_tgt_img = self.postprocess(image=tgt_img)['image']
        return norm_img, (norm_tgt_img, label)
