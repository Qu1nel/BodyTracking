from typing import List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from .base_solution import BaseSolution, Landmark, _normalize_to_pixel_coordinates
from .base_solution import _RGB_CHANNELS

__all__ = ('FaceMesh', 'FacesMeshDetector')


class FaceMesh(BaseSolution):
    __slots__ = (*['landmark_{}'.format(num) for num in range(0, 468)], '__landmarks')

    def __init__(self, source: NormalizedLandmarkList, image: np.ndarray):
        super().__init__()

        image_rows, image_cols, _ = image.shape
        for idx, raw_lnd in enumerate(source.landmark):
            landmark_px = _normalize_to_pixel_coordinates(raw_lnd.x, raw_lnd.y, image_cols, image_rows)
            self.landmarks[idx] = Landmark(*landmark_px)

        self._init_points()

    def _init_points(self):
        for name, value in zip(self.__slots__, self.landmarks.values()):
            setattr(self, name, value)


class FacesMeshDetector(object):
    static_image_mode: bool
    max_num_faces: int
    refine_landmarks: bool
    min_detection_confidence: float
    min_tracking_confidence: float

    __slots__ = '__faces'

    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.__faces = mp.solutions.face_mesh.FaceMesh(
            static_image_mode, max_num_faces, refine_landmarks, min_detection_confidence, min_tracking_confidence
        )

    def process(self, image: np.ndarray) -> List:
        """Converts the image from BGR to RGB, then processes an RGB image and returns the face
            landmarks and handedness of each detected face.Returns the image in BGR after processing.

        Args:
          image: An RGB image represented as a numpy ndarray.

        Raises:
          ValueError: If the input image is not three channel RGB.

        Returns:
          A list object that contains the face landmarks on each detected face.
        """
        if image.shape[2] != _RGB_CHANNELS:
            raise ValueError('Input image must contain three channel rgb data.')

        # writeable = False to improve performance
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_faces = self.__faces.process(image).multi_face_landmarks

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        results = []
        if raw_faces:
            for raw_face in raw_faces:
                results.append(FaceMesh(raw_face, image))

        return results

    def __enter__(self):
        """A "with" statement support."""
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        """A "with" statement support."""
        if exc_type is not None:
            print(exc_type, exc_val, traceback)  # traceback.(tb_frame, tb_lasti, tb_lineno, tb_next)
            return False
        return self
