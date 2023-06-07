from typing import List, Literal

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from src.tracking.bodyparts.base_solution import BaseSolution
from src.tracking.bodyparts.base_solution import LabelMixin
from src.tracking.bodyparts.base_solution import ContextMixin
from src.tracking.bodyparts.base_solution import RGB_CHANNELS

__all__ = ('FaceMesh', 'FacesMeshDetector')


class FaceMesh(BaseSolution, LabelMixin):
    __slots__ = (*[f'landmark_{num}' for num in range(0, 468)], '__landmarks')

    def __init__(self, source: NormalizedLandmarkList, image: np.ndarray, label: Literal['face'] = 'facemesh'):
        BaseSolution.__init__(self, source, image)
        LabelMixin.__init__(self, label)


class FacesMeshDetector(ContextMixin):
    __faces: mp.solutions.face_mesh.FaceMesh

    def __init__(
            self,
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
    ):
        """Init method for FaceMeshDetector.

        Args:
            static_image_mode: Whether to treat the input images as a batch of static
                and possibly unrelated images, or a video stream. See details in
                https://solutions.mediapipe.dev/face_mesh#static_image_mode.

            max_num_faces: Maximum number of faces to detect. See details in
                https://solutions.mediapipe.dev/face_mesh#max_num_faces.

            refine_landmarks: Whether to further refine the landmark coordinates
                around the eyes and lips, and output additional landmarks around the
                irises. Default to False. See details in
                https://solutions.mediapipe.dev/face_mesh#refine_landmarks.

            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful. See details in
                https://solutions.mediapipe.dev/face_mesh#min_detection_confidence.

            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
                face landmarks to be considered tracked successfully. See details in
                https://solutions.mediapipe.dev/face_mesh#min_tracking_confidence.
        """
        self.__faces = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence

        )

    def process(self, image: np.ndarray) -> List[FaceMesh]:
        """Converts the image from BGR to RGB, then processes an RGB image and returns the face landmarks.

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

        raw_faces = self.__faces.process(image).multi_face_landmarks

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        results = []
        if raw_faces:
            for raw_face in raw_faces:
                results.append(FaceMesh(raw_face, image))

        return results
