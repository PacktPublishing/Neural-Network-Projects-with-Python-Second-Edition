import data_points
import numpy as np


def test_get_points():
    x, y = data_points.get_points(count=10)
    assert x.shape == (10, )
    assert y.shape == (10, )


def test_mse():
    r = data_points.calc_mse(np.array([1, 1, 1]), np.array([1, 1, 0]))
    assert abs(r - 1/3) < 1e-10
    r = data_points.calc_mse(np.array([0, 1, 1]), np.array([2, 1, 0]))
    assert abs(r - 5/3) < 1e-10


def test_mae():
    r = data_points.calc_mae(np.array([1, 1, 1]), np.array([1, 1, 0]))
    assert abs(r - 1/3) < 1e-10
    r = data_points.calc_mae(np.array([0, 1, 1]), np.array([2, 1, 0]))
    assert abs(r - 1.0) < 1e-10
