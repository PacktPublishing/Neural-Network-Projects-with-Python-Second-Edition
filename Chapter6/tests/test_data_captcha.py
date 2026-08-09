import data_captcha
import torch


def test_get_samples():
    p = data_captcha.get_images_path()
    res = data_captcha.get_samples(p)
    assert len(res) == 113062


def test_generate_ocr_val_set():
    r = data_captcha.generate_ocr_val_set(size=10, seed=123)
    assert torch.is_tensor(r[1])
    assert r[1].shape == (10, 1, data_captcha.CELL_HEIGHT, data_captcha.CELL_WIDTH)
