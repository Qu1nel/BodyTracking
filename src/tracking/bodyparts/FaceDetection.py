from typing import List, Literal

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.location_data_pb2 import LocationData

from src.tracking.bodyparts.base_solution import BaseSolution
from src.tracking.bodyparts.base_solution import ContextMixin
from src.tracking.bodyparts.base_solution import LabelMixin
from src.tracking.bodyparts.base_solution import RGB_CHANNELS

__all__ = ('Face', 'FacesDetector')


class Face(BaseSolution, LabelMixin):
    __slots__ = (
        'landmark_0', 'landmark_1', 'landmark_2', 'landmark_3',
        'landmark_4', 'landmark_5', 'landmark_6', '__landmarks',
        'relative_bounding_box', 'name'
    )

    def __init__(self, source: LocationData, image: np.ndarray, label: Literal['face'] = 'face'):
        BaseSolution.__init__(self, source, image)
        LabelMixin.__init__(self, label)

        self.relative_bounding_box = dict(
            xmin=source.relative_bounding_box.xmin,
            ymin=source.relative_bounding_box.ymin,
            width=source.relative_bounding_box.width,
            height=source.relative_bounding_box.height
        )


class FacesDetector(ContextMixin):
    __face: mp.solutions.face_detection.FaceDetection

    def __init__(
            self,
            min_detection_confidence=0.5,
            model_selection=0,
    ):
        """Init method for FaceDetector.

        Args:
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful. See details in
                https://solutions.mediapipe.dev/face_detection#min_detection_confidence.

            model_selection: 0 or 1. 0 to select a short-range model that works
                best for faces within 2 meters from the camera, and 1 for a full-range
                model best for faces within 5 meters. See details in
                https://solutions.mediapipe.dev/face_detection#model_selection.
        """
        self.__face = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )

    def process(self, image: np.ndarray) -> List[Face]:
        """Converts the image from BGR to RGB, then processes an RGB image and
        returns the detected face location data.

        Returns the image in BGR after processing.

        Args:
            image: An RGB image represented as a numpy ndarray.

        Raise:
            ValueError: If the input image is not three channel RGB.

        Returns:
            A list object that contains the face object
        """

        if image.shape[2] != RGB_CHANNELS:
            raise ValueError('Input image must contain three channel rgb data.')

        # writeable = False to improve performance
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_faces = self.__face.process(image).detections

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        results = []
        if raw_faces:
            for raw_face in raw_faces:
                results.append(Face(raw_face.location_data, image))

        return results
