from typing import List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from .base_solution import Landmark, BaseSolution, _normalize_to_pixel_coordinates
from .base_solution import _RGB_CHANNELS

__all__ = ('Hand', 'HandsDetector')


class Hand(BaseSolution):
    __slots__ = (
        'landmark_0', 'landmark_1', 'landmark_2', 'landmark_3',
        'landmark_4', 'landmark_5', 'landmark_6', 'landmark_7',
        'landmark_8', 'landmark_9', 'landmark_10', 'landmark_11',
        'landmark_12', 'landmark_13', 'landmark_14', 'landmark_15',
        'landmark_16', 'landmark_17', 'landmark_18', 'landmark_19',
        'landmark_20', '__landmarks'
    )

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


class HandsDetector(object):
    max_num_hands: int
    min_tracking_confidence: float
    min_detection_confidence: float

    __slots__ = '__hand'

    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Args:
          static_image_mode: Whether to treat the input images as a batch of static
            and possibly unrelated images, or a video stream. See details in
            https://solutions.mediapipe.dev/hands#static_image_mode.
          max_num_hands: Maximum number of hands to detect. See details in
            https://solutions.mediapipe.dev/hands#max_num_hands.
          min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand
            detection to be considered successful. See details in
            https://solutions.mediapipe.dev/hands#min_detection_confidence.
          min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
            hand landmarks to be considered tracked successfully. See details in
            https://solutions.mediapipe.dev/hands#min_tracking_confidence.
        """
        self.__hand = mp.solutions.hands.Hands(max_num_hands=max_num_hands,
                                               min_detection_confidence=min_detection_confidence,
                                               min_tracking_confidence=min_tracking_confidence)

    def process(self, image: np.ndarray) -> List:
        """Converts the image from BGR to RGB, then processes an RGB image and returns the hand
            landmarks and handedness of each detected hand.Returns the image in BGR after processing.

        Args:
          image: An RGB image represented as a numpy ndarray.

        Raise:
          ValueError: If the input image is not three channel RGB.

        Returns:
          A list object that contains the hand landmarks on each detected hand.
        """
        if image.shape[2] != _RGB_CHANNELS:
            raise ValueError('Input image must contain three channel rgb data.')

        # writeable = False to improve performance
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_hands = self.__hand.process(image).multi_hand_landmarks

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        results = []
        if raw_hands:
            for raw_hand in raw_hands:
                results.append(Hand(raw_hand, image))

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
