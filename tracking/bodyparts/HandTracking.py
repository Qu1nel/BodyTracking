from typing import List, Literal

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from src.tracking.bodyparts.base_solution import BaseSolution
from src.tracking.bodyparts.base_solution import LabelMixin
from src.tracking.bodyparts.base_solution import ContextMixin
from src.tracking.bodyparts.base_solution import RGB_CHANNELS

__all__ = ('Hand', 'HandsDetector')


class Hand(BaseSolution, LabelMixin):
    __slots__ = (
        'landmark_0', 'landmark_1', 'landmark_2', 'landmark_3',
        'landmark_4', 'landmark_5', 'landmark_6', 'landmark_7',
        'landmark_8', 'landmark_9', 'landmark_10', 'landmark_11',
        'landmark_12', 'landmark_13', 'landmark_14', 'landmark_15',
        'landmark_16', 'landmark_17', 'landmark_18', 'landmark_19',
        'landmark_20', '__landmarks', 'name'
    )

    def __init__(self, source: NormalizedLandmarkList, image: np.ndarray, label: Literal['face'] = 'hand'):
        BaseSolution.__init__(self, source, image)
        LabelMixin.__init__(self, label)


class HandsDetector(ContextMixin):
    __hand: mp.solutions.hands.Hands

    def __init__(
            self,
            max_num_hands: int = 2,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5,
    ):
        """Init method for HandDetector.

        Args:
            max_num_hands: Maximum number of hands to detect. See details in
                https://solutions.mediapipe.dev/hands#max_num_hands.

            min_detection_confidence: Minimum confidence value ([0.0, 1.0])
                for hand detection to be considered successful. See details in
                https://solutions.mediapipe.dev/hands#min_detection_confidence.

            min_tracking_confidence: Minimum confidence value ([0.0, 1.0])
                for the hand landmarks to be considered tracked successfully.
                See details in https://solutions.mediapipe.dev/hands#min_tracking_confidence.
        """
        self.__hand = mp.solutions.hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, image: np.ndarray) -> List[Hand]:
        """Converts the image from BGR to RGB, then processes an RGB image and returns the hand landmarks.

        Returns the image in BGR after processing.

        Args:
            image: An RGB image represented as a numpy ndarray.

        Raise:
            ValueError: If the input image is not three channel RGB.

        Returns:
            A list object that contains the hand landmarks on each detected hand.
        """

        if image.shape[2] != RGB_CHANNELS:
            raise ValueError('Input image must contain 3 channels RGB data.')

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
