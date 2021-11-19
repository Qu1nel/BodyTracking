from typing import List

import mediapipe as mp
import numpy as np

from .base_solution import BaseSolution
from .base_solution import _RGB_CHANNELS

__all__ = ('FaceMesh', 'FacesMeshDetector')


class FaceMesh(BaseSolution):
    def _init_points(self):
        pass


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

        raw_faces = self.__faces.process(image).multi_hand_landmarks

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
