import data_captcha


def test_get_samples():
    p = data_captcha.get_images_path()
    res = data_captcha.get_samples(p)
    assert len(res) == 113062

