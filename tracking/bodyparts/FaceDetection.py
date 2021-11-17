from typing import List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.location_data_pb2 import LocationData

from .base_solution import BaseSolution, _normalize_to_pixel_coordinates, Landmark


class Face(BaseSolution):
    __slots__ = (
        'landmark_0', 'landmark_1', 'landmark_2', 'landmark_3',
        'landmark_4', 'landmark_5', 'landmark_6', '__landmarks',
        'relative_bounding_box'
    )

    def __init__(self, source: LocationData, image: np.ndarray):
        super().__init__()

        image_rows, image_cols, _ = image.shape
        for idx, raw_lnd in enumerate(source.relative_keypoints):
            landmark_px = _normalize_to_pixel_coordinates(raw_lnd.x, raw_lnd.y, image_cols, image_rows)
            self.landmarks[idx] = Landmark(*landmark_px)

        self._init_points()

        self.relative_bounding_box = dict(
            xmin=source.relative_bounding_box.xmin,
            ymin=source.relative_bounding_box.ymin,
            width=source.relative_bounding_box.width,
            height=source.relative_bounding_box.height
        )

    def _init_points(self):
        for name, value in zip(self.__slots__, self.landmarks.values()):
            setattr(self, name, value)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        part = '\n\txmin:{}\n\tymin:{}\n\twidth:{}\n\theight:{}\n\n'. \
            format(self.relative_bounding_box['xmin'], self.relative_bounding_box['ymin'],
                   self.relative_bounding_box['width'], self.relative_bounding_box['height'])

        return 'relative_bounding_box:' + part + 'landmarks:\n' + '\t{}\n'.format(self.landmarks)


class FacesDetector(object):
    min_detection_confidence: float
    model_selection: int
    __slots__ = '__face'

    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """Initializes a MediaPipe Face Detection object.

        Args:
          min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
            detection to be considered successful. See details in
            https://solutions.mediapipe.dev/face_detection#min_detection_confidence.
          model_selection: 0 or 1. 0 to select a short-range model that works
            best for faces within 2 meters from the camera, and 1 for a full-range
            model best for faces within 5 meters. See details in
            https://solutions.mediapipe.dev/face_detection#model_selection.
        """
        self.__face = mp.solutions.face_detection.FaceDetection(min_detection_confidence, model_selection)

    def process(self, image: np.ndarray) -> List:
        """Converts the image from BGR to RGB, then processes an RGB image and returns
            the detected face location data. Returns the image in BGR after processing.

        Args:
          image: An RGB image represented as a numpy ndarray.

        Raises:
          RuntimeError: If the underlying graph throws any error.
          ValueError: If the input image is not three channel RGB.

        Returns:
            A list object that contains the face object
        """
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

    def __enter__(self):
        """A "with" statement support."""
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        """A "with" statement support."""
        if exc_type is not None:
            print(exc_type, exc_val, traceback)  # traceback.(tb_frame, tb_lasti, tb_lineno, tb_next)
            return False
        return self
