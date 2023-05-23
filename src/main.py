import cv2

from tracking import *

camera = cv2.VideoCapture(0)
with FaceMeshDetection.FacesMeshDetector() as Face, HandTracking.HandsDetector(2) as Hand:
    while camera.isOpened():
        status, image = camera.read()
        if not status:
            print("Ignoring empty camera frame.")
            break

        results_faces = Face.process(image)
        results_hands = Hand.process(image)

        if results_hands:
            for hand in results_hands:
                DrawingUtils.draw_landmarks(image, hand, connections=connections.HandConnections.HAND_CONNECTIONS_V1)

        if results_faces:
            for face in results_faces:
                DrawingUtils.draw_landmarks(image, face, connections=connections.FaceConnections.FACEMESH_TESSELATION)

        cv2.imshow(name_window := 'SOME NAME', image)

        if cv2.waitKey(1) == 27 or not cv2.getWindowProperty(name_window, cv2.WND_PROP_VISIBLE):
            break

cv2.destroyAllWindows()
