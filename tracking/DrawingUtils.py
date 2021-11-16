import dataclasses
import math
from typing import Tuple, Union, Mapping, List, Optional

import cv2
import numpy as np

_RGB_CHANNELS = 3

WHITE = (224, 224, 224)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


@dataclasses.dataclass
class DrawingSpec(object):
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE
    # Thickness for drawing the annotation. Default to 1 pixel.
    thickness: int = 1
    # Circle radius. Default to 1 pixel.
    circle_radius: int = 1


def _normal_to_pix_coord(normal_x: float, normal_y: float, image_w: int, image_h: int) -> Tuple[int, int]:
    """Converts normalized value pair to pixel coordinates."""
    x_px = min(math.ceil(normal_x * image_w), image_w - 1)
    y_px = min(math.ceil(normal_y * image_h), image_h - 1)
    return x_px, y_px


def draw_landmarks(
        image: np.ndarray,
        landmark_list: Mapping,
        connections: Optional[List[Tuple[int, int]]] = None,
        landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(color=RED, thickness=2),
        connection_drawing_spec: Union[DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec(color=GREEN),
        ignore_landmark: Union[List[int], Tuple[int]] = tuple()):
    """Draws the landmarks and the connections on the image.

    Args:
      image: A three channel RGB image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on
        the image.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected in the drawing.
      landmark_drawing_spec: Either a DrawingSpec object or a mapping from
        hand landmarks to the DrawingSpecs that specifies the landmarks' drawing
        settings such as color, line thickness, and circle radius.
        If this argument is explicitly set to None, no landmarks will be drawn.
      connection_drawing_spec: Either a DrawingSpec object or a mapping from
        hand connections to the DrawingSpecs that specifies the
        connections' drawing settings such as color and line thickness.
        If this argument is explicitly set to None, no landmark connections will
        be drawn.
      ignore_landmark: A sequence of landmarks that will be skipped during drawing

    Raises:
      ValueError: If one of the followings:
        a) If the input image is not three channel RGB.
        b) If any connetions contain invalid landmark index.
    """
    if not landmark_list:
        return None
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')

    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = landmark_list.landmarks

    if connections:
        num_landmarks = len(landmark_list.landmarks)

        for connection in connections:
            start_idx, end_idx = connection[0], connection[1]

            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')

            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                if isinstance(connection_drawing_spec, Mapping):
                    drawing_spec = connection_drawing_spec[connection]
                else:
                    drawing_spec = connection_drawing_spec

                cv2.line(image, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx],
                         drawing_spec.color, drawing_spec.thickness)

    if landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            if idx in ignore_landmark:
                continue

            if isinstance(landmark_drawing_spec, Mapping):
                drawing_spec = landmark_drawing_spec[idx]
            else:
                drawing_spec = landmark_drawing_spec

            cv2.circle(image, landmark_px, drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)
