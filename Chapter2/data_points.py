# this is a very simple data we use to train our neural network using various libraries
import numpy as np
import typing as tt


MIN_X = 0.0
MAX_X = 10.0
POINTS_COUNT = 100
TRUE_SLOPE = 0.4
TRUE_INTERCEPT = 0.1


def get_points(count: int = POINTS_COUNT, min_x: float = MIN_X,
               max_x: float = MAX_X, slope: float = TRUE_SLOPE,
               intercept: float = TRUE_INTERCEPT,
               seed: int = 42) -> tt.Tuple[np.ndarray, np.ndarray]:
    """
    Generate random (but reproducible) points on the plane along
    the given line
    :param count: how many points to generate
    :param min_x: minimum value of X
    :param max_x: maximum value of X
    :param slope: slope of the line
    :param intercept: intercept of the line
    :param seed: random seed
    :return: tuple with x and y coordinates
    """
    x = np.arange(min_x, max_x, (max_x - min_x) / count)
    np.random.seed(seed)
    noise = np.random.uniform(-.5, .5, size=len(x))
    y = x * slope + intercept + noise
    return x, y


def calc_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Square Error
    :param y_true: true values
    :param y_pred: predicted values
    :return: error value as float
    """
    return np.mean((y_true - y_pred) ** 2).item()


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error
    :param y_true: true values
    :param y_pred: predicted values
    :return: error value as float
    """
    return np.mean(np.abs(y_true - y_pred)).item()
