import math

from typing import Tuple, Literal, Union

import numpy as np

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.framework.formats.location_data_pb2 import LocationData

RGB_CHANNELS = 3


class BaseSolution:
    __landmarks: dict

    def __init__(self, source: Union[NormalizedLandmarkList, LocationData], image: np.ndarray):
        self.__landmarks = dict()

        if isinstance(source, NormalizedLandmarkList):
            mask = source.landmark
        elif isinstance(source, LocationData):
            mask = source.relative_keypoints
        else:
            raise TypeError(f"Arg `source` is must be NormalizedLandmarkList or LocationData, not ({type(source)})")

        image_rows, image_cols, _ = image.shape
        for idx, raw_lnd in enumerate(mask):
            landmark_px = normalize_to_pixel_coordinates(
                normal_x=raw_lnd.x,
                normal_y=raw_lnd.y,
                image_w=image_cols,
                image_h=image_rows
            )
            x, y = landmark_px
            self.landmarks[idx] = Landmark(x, y)  # self.landmarks is dict

        self._init_points()

    def _init_points(self):
        for name, value in zip(self.__slots__, self.landmarks.values()):
            setattr(self, name, value)

    @property
    def landmarks(self):
        return self.__landmarks


class LabelMixin:
    name: Literal['hand', 'facemesh', 'face', 'pose']

    def __init__(self, name: Literal['hand', 'facemesh', 'face', 'pose']):
        self.name = name


class ContextMixin:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type is not None:
            print(exc_type, exc_val, traceback)  # traceback.(tb_frame, tb_lasti, tb_lineno, tb_next)
            return False


class Landmark:
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
        return self.__repr__()

    def __repr__(self):
        return 'Landmark(x={} y={})'.format(self.x, self.y)


def normalize_to_pixel_coordinates(normal_x: float, normal_y: float, image_w: int, image_h: int) -> Tuple[int, int]:
    """Converts normalized value pair to pixel coordinates."""
    x_px = min(math.ceil(normal_x * image_w), image_w - 1)
    y_px = min(math.ceil(normal_y * image_h), image_h - 1)

    return x_px, y_px
