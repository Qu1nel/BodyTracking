from typing import Any, Union, List

import cv2

import tracking.connections as connections

from tracking import DrawingUtils
from tracking.bodyparts.HandTracking import HandsDetector, Hand
from tracking.bodyparts.FaceMeshDetection import FacesMeshDetector, FaceMesh
from tracking.bodyparts.FaceDetection import FacesDetector, Face
from tracking.bodyparts.base_solution import LabelMixin

TITLE_WINDOW = 'Tracking'
DEFAULT_HAND_CONNECTIONS = connections.HandConnections.HAND_CONNECTIONS_V2
DEFAULT_FACEMESH_TESSELATION = connections.FaceConnections.FACEMESH_TESSELATION


def pressed_exit() -> bool:
    return cv2.waitKey(1) == 27 or not cv2.getWindowProperty(TITLE_WINDOW, cv2.WND_PROP_VISIBLE)


def status_good(status: Any) -> bool:
    return bool(status)


def main(source) -> None:
    with HandsDetector(min_detection_confidence=0.7, min_tracking_confidence=0.2, max_num_hands=2) as HandObj, \
            FacesMeshDetector() as FaceMeshObj, FacesDetector() as FaceObj:

        handler_objects = [HandObj, FaceObj, FaceMeshObj]

        while source.isOpened():
            status, image = camera.read()

            if not status_good(status):
                print("Ignoring empty camera frame.")
                break

            result_processing: List[Union[Hand, Face, FaceMesh, LabelMixin]] = [
                entity for obj in handler_objects for entity in obj.process(image)
            ]

            for result in result_processing:
                label = result.name

                if label in ('hand', 'facemesh'):
                    if label == 'hand':
                        connection = DEFAULT_HAND_CONNECTIONS
                    elif label == 'facemesh':
                        connection = DEFAULT_FACEMESH_TESSELATION

                    DrawingUtils.draw_landmarks(image, result, connections=connection)

                elif label == 'face':
                    DrawingUtils.draw_detection(image, result)
                else:
                    raise RuntimeError

            cv2.imshow(TITLE_WINDOW, image)

            if pressed_exit():
                break


if __name__ == '__main__':
    try:
        camera = cv2.VideoCapture(0)
        main(source=camera)
    except Exception as exc:
        import sys

        line_number = exc.__traceback__.tb_lineno
        file = exc.__traceback__.tb_frame.f_code.co_filename
        error = exc.__repr__()

        sys.stderr.write(
            "An unexpected error has occurred! Details:\n"
            f"\tfile: {file}\n"
            f"\tline: {line_number}\n"
            f"\terror: {error}\n\n"
        )

        raise exc from None
    finally:
        cv2.destroyAllWindows()
