import cv2
import numpy as np
from typing import Union, Optional
import mediapipe as mp
from typing import List, Tuple, Optional

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the specified file path.

    Args:
    image_path: The file path to the image.

    Returns:
    A numpy ndarray representing the loaded image.

    Raises:
    FileNotFoundError: If the image file does not exist.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Converts an image to the RGB color space.

    This function handles different types of images, including those in grayscale
    and floating-point format, as well as standard BGR and BGRA images.

    Args:
    image: A numpy ndarray representing the loaded image.

    Returns:
    A numpy ndarray of the image in RGB format.

    Raises:
    ValueError: If the image is in an unsupported format.
    """
    # If the image is in floating-point format, convert to 8-bit integer
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image, 0, 1)  # Ensure values are within [0, 1] range
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 2:
        # Image is grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Image is BGR
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Image is BGRA
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError("Unsupported image format")


def process_image(image_path: str) -> np.ndarray:
    """
    Loads an image from a file path and converts it to RGB color space.

    Args:
    image_path: The file path to the image.

    Returns:
    A preprocessed image in RGB format.

    Raises:
    FileNotFoundError: If the image file does not exist.
    ValueError: If the image is in an unsupported format.
    """
    image = load_image(image_path)
    image_rgb = convert_to_rgb(image)

    # Resize and preprocess (if necessary)
    # Example: Resize to 256x256. Modify as per your requirements.
    resized_image = cv2.resize(image_rgb, (256, 256))

    return resized_image



def detect_faces(image):
    """
    Detects faces in an image using MediaPipe Face Detection.

    Args:
    image: An RGB image array.

    Returns:
    Annotated image with face detections.
    """

    # Initialize MediaPipe Face Detection.
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # Process the image and detect faces.
        results = face_detection.process(image)

        # Draw face detections on the image.
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

    return image



def detect_faces_borders(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detects faces in an image using MediaPipe Face Detection and returns the face borders.

    Args:
    image: An RGB image array.

    Returns:
    A list of bounding boxes for each detected face, represented as tuples (x, y, width, height).
    """
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # Process the image and detect faces.
        results = face_detection.process(image)

        # Extract bounding boxes for each detection
        face_borders = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                face_borders.append((x, y, w, h))

    return face_borders

