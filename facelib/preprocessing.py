import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the specified file path.

    Args:
    image_path (str): The file path to the image.

    Returns:
    np.ndarray: A numpy ndarray representing the loaded image.

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
    image (np.ndarray): A numpy ndarray representing the loaded image.

    Returns:
    np.ndarray: A numpy ndarray of the image in RGB format.

    Raises:
    ValueError: If the image is in an unsupported format.
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image, 0, 1)  # Ensure values are within [0, 1] range
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] in [3, 4]:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB if image.shape[2] == 3 else cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError("Unsupported image format")


def adjust_brightness_contrast(image: np.ndarray, brightness=0, contrast=0) -> np.ndarray:
    """
    Adjusts the brightness and contrast of an image.

    Args:
    image (np.ndarray): The input image array.
    brightness (int): The brightness adjustment factor.
    contrast (int): The contrast adjustment factor.

    Returns:
    np.ndarray: The adjusted image array.
    """
    adjusted = np.clip((1.0 + contrast / 100.0) * image - contrast / 100.0 + brightness, 0, 255).astype(np.uint8)
    return adjusted


def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
    """
    Applies Gaussian blur to an image.

    Args:
    image (np.ndarray): The input image array.
    kernel_size (Tuple[int, int]): The size of the Gaussian kernel.

    Returns:
    np.ndarray: The blurred image array.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes an image by scaling pixel values to the range [0, 1].

    Args:
    image (np.ndarray): The input image array.

    Returns:
    np.ndarray: The normalized image array.
    """
    return image.astype('float32') / 255.0


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Applies histogram equalization to enhance image contrast.

    Args:
    image (np.ndarray): The input grayscale or RGB image array.

    Returns:
    np.ndarray: The image with equalized histogram.
    """
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to YUV and equalize the Y channel
        yuv_img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
    else:
        raise ValueError("Unsupported image format for histogram equalization")


def resize_with_aspect_ratio(image: np.ndarray, target_height: int) -> np.ndarray:
    """
    Resizes an image while preserving its aspect ratio.

    Args:
    image (np.ndarray): The input image array.
    target_height (int): The target height for the resized image.

    Returns:
    np.ndarray: The resized image array.
    """
    h, w = image.shape[:2]
    scaling_factor = target_height / h
    new_width = int(w * scaling_factor)
    return cv2.resize(image, (new_width, target_height))


def process_image(image_path: str, target_height: int = 256, preserve_aspect_ratio: bool = True) -> np.ndarray:
    """
    Loads an image from a file path, converts it to RGB, and resizes it while optionally preserving its aspect ratio.

    Args:
    image_path (str): The file path to the image.
    target_height (int): The target height for the resized image.
    preserve_aspect_ratio (bool): If True, preserves the image's aspect ratio during resizing.

    Returns:
    np.ndarray: A preprocessed image in RGB format.

    Raises:
    FileNotFoundError: If the image file does not exist.
    ValueError: If the image is in an unsupported format.
    """
    image = load_image(image_path)
    image_rgb = convert_to_rgb(image)

    if preserve_aspect_ratio:
        resized_image = resize_with_aspect_ratio(image_rgb, target_height)
    else:
        resized_image = cv2.resize(image_rgb,
                                   (target_height, target_height))  # Square resize if aspect ratio is not preserved

    return resized_image


def detect_faces(image: np.ndarray) -> np.ndarray:
    """
    Detects faces in an image using MediaPipe Face Detection and annotates them.

    Args:
    image (np.ndarray): An RGB image array.

    Returns:
    np.ndarray: The image annotated with face detections.
    """
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

    return image


def detect_faces_borders(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detects faces in an image using MediaPipe Face Detection and returns their borders.

    Args:
    image (np.ndarray): An RGB image array.

    Returns:
    List[Tuple[int, int, int, int]]: A list of bounding boxes for each detected face,
     formatted as (x, y, width, height).
    """
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image)
        face_borders = []

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_borders.append((x, y, w, h))

    return face_borders
