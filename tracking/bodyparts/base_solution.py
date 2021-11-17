import math
from abc import abstractmethod
from typing import Tuple


class BaseSolution(object):
    __landmarks: dict

    def __init__(self):
        self.__landmarks = dict()

    @abstractmethod
    def _init_points(self):
        pass

    @property
    def landmarks(self):
        return self.__landmarks


class Landmark(object):
    __slots__ = ('x', 'y', '__sequence')

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.__sequence = (self.x, self.y)

    def __getitem__(self, key):
        return self.__sequence[key]

    def __len__(self):
        return len(self.__sequence)

    def __str__(self):
        return 'Landmark(x={} y={})'.format(self.x, self.y)

    def __repr__(self):
        return 'Landmark(x={} y={})'.format(self.x, self.y)


def _normalize_to_pixel_coordinates(normal_x: float, normal_y: float, image_w: int, image_h: int) -> Tuple[int, int]:
    """Converts normalized value pair to pixel coordinates."""
    x_px = min(math.ceil(normal_x * image_w), image_w - 1)
    y_px = min(math.ceil(normal_y * image_h), image_h - 1)
    return x_px, y_px
