# Functions related to captcha dataset
import kagglehub as kh
import cv2
import random
import typing as tt
import numpy as np
import pathlib
import string
import itertools
import torch
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


def generate_ocr_val_set(size: int, seed: int) -> tt.Tuple[tt.List[str], torch.Tensor]:
    """
    Generate validation set --- blurred images from random characters
    :return: random chars and their images as single tensor
    """
    random.seed(seed)
    blur_pipeline = A.GaussianBlur(p=0.5)
    captchas = []
    arrays = []
    for _ in range(size):
        s = random.choice(CAPTCHA_CHARS)
        captchas.append(s)
        img = make_target_image(s)
        img_np = np.array(img)
        img_np = (img_np.astype(np.float32) / 255.0)
        blur_img = blur_pipeline(image=img_np)['image']
        arrays.append(np.expand_dims(blur_img, 0))
    return captchas, torch.as_tensor(np.stack(arrays))


class OCRDataset(Dataset):
    def __init__(self, size_mul: int = 1):
        self.size_mul = size_mul
        self.postprocess = A.Compose([
            A.GaussianBlur(p=0.5),
            A.ToTensorV2(),
        ])

    def __len__(self):
        return len(CAPTCHA_CHARS) * self.size_mul

    def __getitem__(self, item):
        item %= len(CAPTCHA_CHARS)
        img = make_target_image(CAPTCHA_CHARS[item])
        img_np = np.array(img)
        img_np = (img_np.astype(np.float32) / 255.0)
        res = self.postprocess(image=img_np)['image']
        return res, item



class CaptchaDataset(Dataset):
    def __init__(self, samples: tt.List[TSample]):
        self.samples = samples
        self.postprocess = A.Compose([
            A.ToTensorV2(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        file_path, label = self.samples[item]
        img = cv2.imread(file_path, flags=cv2.IMREAD_COLOR_RGB)
        img = (img.astype(np.float32) / 255.0)
        norm_img = self.postprocess(image=img)['image']
        tgt_img = np.array(make_target_image(label))
        tgt_img = (tgt_img.astype(np.float32) / 255.0)
        norm_tgt_img = self.postprocess(image=tgt_img)['image']
        return norm_img, (norm_tgt_img, label)
