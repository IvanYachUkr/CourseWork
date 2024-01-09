import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.9)


def is_image_blurry(image: np.ndarray, threshold: float = 60.0) -> bool:
    """
    Checks if the image is blurry based on the variance of the Laplacian.

    Args:
    image (np.ndarray): The input image in BGR format.
    threshold (float): The threshold for determining if the image is blurry.

    Returns:
    bool: True if the image is blurry, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian < threshold


def face_detected_in_frame(image: np.ndarray, confidence_threshold: float = 0.7) -> bool:
    """
    Detects if there is a face in the given image frame.

    Args:
    image (np.ndarray): The input image frame in BGR format.
    confidence_threshold (float): The confidence threshold for face detection.

    Returns:
    bool: True if a face is detected, False otherwise.
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            if detection.score[0] > confidence_threshold:
                return True
    return False


def capture_frame_with_face() -> Optional[np.ndarray]:
    """
    Captures a video frame from the default camera that contains a face and is not blurry.

    Returns:
    Optional[np.ndarray]: The captured frame as a NumPy array if a suitable frame is found, None otherwise.
    """
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        if face_detected_in_frame(frame) and not is_image_blurry(frame):
            cap.release()
            return np.array(frame)

        cv2.waitKey(1)

    cap.release()
    return None
