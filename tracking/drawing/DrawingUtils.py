from dataclasses import dataclass
from typing import Tuple, Union, Mapping, List, Optional

import cv2
import numpy as np

from tracking.drawing.colors import *
from tracking.bodyparts import base_solution
from tracking.bodyparts.FaceDetection import Face

__all__ = (
    'DrawingSpec',
    'draw_landmarks'
)


@dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE
    # Thickness for drawing the annotation. Default to 1 pixel.
    thickness: int = 1
    # Circle radius. Default to 1 pixel.
    circle_radius: int = 1


def draw_landmarks(
        image: np.ndarray,
        landmark_list: base_solution.BaseSolution,
        connections: Optional[List[Tuple[int, int]]] = None,
        landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(
            color=RED,
            thickness=-1
        ),
        connection_drawing_spec: Union[DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec(
            color=GREEN
        ),
        ignore_landmark: Union[List[int], Tuple[int]] = tuple(),
        *,
        smart_drawing: bool = False
):
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

        ignore_landmark: A sequence of landmarks that will be skipped during drawing.

        smart_drawing: A flag that meaning drawing only those landmarks that are in the
            sequence connections.

    Raises:
        ValueError: If one of the followings:
            a) If the input image is not 3 channel RGB.
            b) If any connections contain invalid landmark index.
    """

    if not landmark_list:
        return None
    if image.shape[2] != base_solution.RGB_CHANNELS:
        raise ValueError('Input image must contain 3 channel rgb data.')

    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = landmark_list.landmarks
    num_landmarks = len(landmark_list.landmarks)

    if connections:
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
        if smart_drawing:
            ignore_landmark = [idx for idx in range(0, num_landmarks) if idx not in sum(connections, ())]

        for idx, landmark_px in idx_to_coordinates.items():
            if idx in ignore_landmark:
                continue

            if isinstance(landmark_drawing_spec, Mapping):
                drawing_spec = landmark_drawing_spec[idx]
            else:
                drawing_spec = landmark_drawing_spec

            cv2.circle(image, landmark_px, drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)


def draw_detection(
        image: np.ndarray,
        detection: Face,
        landmark_drawing_spec: DrawingSpec = DrawingSpec(color=BLUE, circle_radius=3, thickness=-1),
        bbox_drawing_spec: DrawingSpec = DrawingSpec(color=PURPLE),
        *,
        draw_landmark: bool = True,

):
    """Draws the detection bounding box and landmarks on the image.

    Args:
        image: A three channel RGB image represented as numpy ndarray.

        detection: A detection proto message to be annotated on the image.

        landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
            drawing settings such as color, line thickness, and circle radius.

        bbox_drawing_spec: A DrawingSpec object that specifies the bounding box's
            drawing settings such as color and line thickness.

        draw_landmark: Flag indicating whether to draw dots on the image

    Raises:
        ValueError: If the input image is not 3 channel RGB.
    """
    if image.shape[2] != base_solution.RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')

    image_rows, image_cols, _ = image.shape

    if draw_landmark:
        for landmark in detection.landmarks.values():
            cv2.circle(image, landmark, landmark_drawing_spec.circle_radius,
                       landmark_drawing_spec.color, landmark_drawing_spec.thickness)

    # Draws bounding box if exists.
    box = detection.relative_bounding_box
    rect_start_point = base_solution.normalize_to_pixel_coordinates(
        box['xmin'], box['ymin'], image_cols, image_rows
    )
    rect_end_point = base_solution.normalize_to_pixel_coordinates(
        box['xmin'] + box['width'], box['ymin'] + box['height'], image_cols, image_rows
    )

    cv2.rectangle(image, rect_start_point, rect_end_point, bbox_drawing_spec.color, bbox_drawing_spec.thickness)
