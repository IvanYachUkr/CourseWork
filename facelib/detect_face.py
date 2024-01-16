from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp


def crop_faces(image: np.ndarray, face_borders: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """
    Crops faces from the given image based on the provided face borders.

    Args:
    image (np.ndarray): A numpy ndarray representing the original image.
    face_borders (List[Tuple[int, int, int, int]]): A list of tuples, each representing the border of a detected face.
                                                    Each tuple contains (x, y, width, height).

    Returns:
    List[np.ndarray]: A list of numpy ndarrays, each representing a cropped face image.
    """
    cropped_faces = []
    for border in face_borders:
        x, y, w, h = border
        x, y = max(0, x), max(0, y)
        w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)
        face_image = image[y:y + h, x:x + w]
        cropped_faces.append(face_image)

    return cropped_faces


def display_cropped_faces(cropped_faces: List[np.ndarray]) -> None:
    """
    Displays cropped faces using Matplotlib.

    Args:
    cropped_faces (List[np.ndarray]): A list of numpy ndarrays, each representing a cropped face image.
    """
    if not cropped_faces:
        print("No faces to display.")
        return

    num_faces = len(cropped_faces)
    fig, axes = plt.subplots(1, num_faces, figsize=(5 * num_faces, 5))

    if num_faces == 1:
        axes.imshow(cropped_faces[0], cmap='gray' if cropped_faces[0].ndim == 2 else None)
        axes.axis('off')
    else:
        for ax, face in zip(axes.flat, cropped_faces):
            ax.imshow(face, cmap='gray' if face.ndim == 2 else None)
            ax.axis('off')

    plt.show()


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
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), int(bbox_c.width * iw), \
                             int(bbox_c.height * ih)

                face_borders.append((x, y, w, h))

    return face_borders
