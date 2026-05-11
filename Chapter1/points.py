# Module provide utilities to generate the points for the optimisation example
import numpy as np
import typing as tt


def get_points(slope: float = 0.4, intercept: float = 0.1,
               seed: int = 42) -> tt.Tuple[np.ndarray, np.ndarray]:
    """
    Generate random (but reproducible) points on the plane along given line
    :param slope: slope of the line
    :param intercept: intercept of the line
    :param seed: random seed
    :return: tuple with x and y coordinates
    """
    x = np.arange(0, 10, .1)
    np.random.seed(seed)
    noise = np.random.uniform(-.5, .5, size=len(x))
    y = x * slope + intercept + noise
    return x, y


def calc_dist(x: np.ndarray, y: np.ndarray,
              slope: float, intercept: float) -> np.float32:
    """
    Calculate vertical distance from given points to the given line
    :param x: x coordinates
    :param y: y coordinates
    :param slope: slope of the line
    :param intercept: intercept of the line
    :return: total vertical distance from every point to the line
    """
    # get Y coordinate on the line for every point
    yy = slope * x + intercept
    return np.sum(np.abs(y - yy))
