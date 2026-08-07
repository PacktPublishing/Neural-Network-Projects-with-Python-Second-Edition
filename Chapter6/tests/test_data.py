import data


def test_get_classes():
    path = data.get_images_path()
    res = data.get_classes(path)
    assert len(res) == 10


def test_get_class_paths():
    path = data.get_images_path()
    mucca_path = path / "mucca"
    files = data.get_class_files(mucca_path)
    assert len(files) == 1866
    assert files[1].name == "OIP--2ix__438O7A-yHzROPhGwHaFj.jpeg"


def test_split_train_val_test():
    samples = list(range(100))
    t1, t2, t3 = data.split_train_val_test(samples, shuffle=False)
    assert t1 == list(range(80))
    assert t2 == list(range(80, 90))
    assert t3 == list(range(90, 100))

    t1, t2, t3 = data.split_train_val_test(samples, seed=42)
    assert t3 == [86, 13, 17, 28, 31, 35, 94, 3, 14, 81]
