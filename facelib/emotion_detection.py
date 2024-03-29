from typing import List, Tuple, Optional, Dict
import cv2
import os
import tempfile
from deepface import DeepFace
import numpy as np
from face_recognition import single_input_as_path


@single_input_as_path
def detect_emotions_in_image(image_path):
    """
    Detects emotions in an image file using DeepFace.

    Args:
    image_path (str): Path to the image file.

    Returns:
    dict: A dictionary containing the detected emotions and their probabilities,
          or None if an error occurs.
    """
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=('emotion',))
        return analysis[0]['emotion']
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return None


def detect_emotions_in_frame(frame: np.ndarray) -> Optional[List[Tuple[str, float]]]:
    """
    Detects emotions in a given video frame using DeepFace.

    Args:
    frame (np.ndarray): The video frame to be analyzed.

    Returns: Optional[List[Tuple[str, float]]]: A list of tuples containing the top 3 dominant emotions and their
    probabilities, or None if an error occurs.
    """
    try:
        if frame is None:
            raise ValueError("Received a None frame")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            file_name = tmpfile.name
            cv2.imwrite(file_name, frame_rgb)

        analysis = DeepFace.analyze(file_name, actions=('emotion',), enforce_detection=False)
        print(f'{analysis=}')
        os.remove(file_name)

        emotions = analysis[0]['emotion']

        return emotions

    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return None


def extract_dominant_emotions(emotions: Dict[str, float], top_n: int = 3) -> List[Tuple[str, float]]:
    """
    Extracts the top N dominant emotions from the emotion analysis results.

    Args:
    emotions (Dict[str, float]): The dictionary of emotions and their probabilities.
    top_n (int): The number of top emotions to extract.

    Returns:
    List[Tuple[str, float]]: A list of the top N emotions and their probabilities.
    """
    return sorted(emotions.items(), key=lambda item: item[1], reverse=True)[:top_n]
