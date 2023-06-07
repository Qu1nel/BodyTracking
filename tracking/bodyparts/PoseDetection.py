from typing import List, Literal

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from tracking.bodyparts.base_solution import BaseSolution
from tracking.bodyparts.base_solution import LabelMixin
from tracking.bodyparts.base_solution import ContextMixin
from tracking.bodyparts.base_solution import RGB_CHANNELS

__all__ = ('Pose', 'PoseDetector')


class Pose(BaseSolution, LabelMixin):
    __slots__ = (
        'landmark_0', 'landmark_1', 'landmark_2', 'landmark_3',
        'landmark_4', 'landmark_5', 'landmark_6', 'landmark_7',
        'landmark_8', 'landmark_9', 'landmark_10', 'landmark_11',
        'landmark_12', 'landmark_13', 'landmark_14', 'landmark_15',
        'landmark_16', 'landmark_17', 'landmark_18', 'landmark_19',
        'landmark_20', 'landmark_21', 'landmark_22', 'landmark_23',
        'landmark_24', 'landmark_25', 'landmark_26', 'landmark_27',
        'landmark_28', 'landmark_28', 'landmark_29', 'landmark_30',
        'landmark_31', 'landmark_32', 'landmark_33', '__landmarks', 'name'
    )

    def __init__(self, source: NormalizedLandmarkList, image: np.ndarray, label: Literal['pose'] = 'pose'):
        BaseSolution.__init__(self, source, image)
        LabelMixin.__init__(self, label)


class PoseDetector(ContextMixin):
    __pose: mp.solutions.pose.Pose

    def __init__(
            self,
            static_image_mode: bool = False,
            model_complexity: int = 1,
            smooth_landmarks: bool = True,
            enable_segmentation: bool = False,
            smooth_segmentation: bool = True,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5
    ):
        """
        Args:
            static_image_mode: Whether to treat the input images as a batch of static
                and possibly unrelated images, or a video stream. See details in
                https://solutions.mediapipe.dev/pose#static_image_mode.

            model_complexity: Complexity of the pose landmark model: 0, 1 or 2. See
                details in https://solutions.mediapipe.dev/pose#model_complexity.

            smooth_landmarks: Whether to filter landmarks across different input
                images to reduce jitter. See details in
                https://solutions.mediapipe.dev/pose#smooth_landmarks.

            enable_segmentation: Whether to predict segmentation mask. See details in
                https://solutions.mediapipe.dev/pose#enable_segmentation.

            smooth_segmentation: Whether to filter segmentation across different input
                images to reduce jitter. See details in
                https://solutions.mediapipe.dev/pose#smooth_segmentation.

            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for person
                detection to be considered successful. See details in
                https://solutions.mediapipe.dev/pose#min_detection_confidence.

            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
                pose landmarks to be considered tracked successfully. See details in
                https://solutions.mediapipe.dev/pose#min_tracking_confidence.
        """
        self.__pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence

        )

    def process(self, image: np.ndarray) -> List[Pose]:
        """Converts the image from BGR to RGB, then processes an RGB image and returns the pose landmarks.

        Returns the image in BGR after processing.

        Args:
            image: An RGB image represented as a numpy ndarray.

        Raises:
            ValueError: If the input image is not three channel RGB.

        Returns:
            A list object that contains the face landmarks on each detected face.
        """

        if image.shape[2] != RGB_CHANNELS:
            raise ValueError('Input image must contain three channel rgb data.')

        # writeable = False to improve performance
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_poses = self.__pose.process(image).pose_landmarks

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        results = []
        if raw_poses:
            results.append(Pose(raw_poses, image))

        return results
