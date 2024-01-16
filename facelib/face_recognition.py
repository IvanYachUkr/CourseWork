import os
import cv2
import numpy as np
from deepface import DeepFace
from functools import wraps
import tempfile
from typing import Tuple, Callable, Any


def input_as_path(func: Callable) -> Callable:
    """
    Decorator to handle inputs as file paths or numpy arrays. If an input is a numpy array,
    it saves the image as a temporary file and then uses its path.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """

    @wraps(func)
    def wrapper(input1: Any, input2: Any, *args, **kwargs) -> Any:
        temp_files = []

        try:
            input1_path = _process_input(input1, temp_files)
            input2_path = _process_input(input2, temp_files)

            return func(input1_path, input2_path, *args, **kwargs)

        finally:
            _cleanup_temp_files(temp_files)

    return wrapper


def _process_input(input_data: Any, temp_files: list) -> str:
    """
    Process a single input. If it's a numpy array, convert it to a temporary file path.

    Args:
        input_data (Any): The input data to be processed.
        temp_files (list): A list to store the paths of temporary files.

    Returns:
        str: The file path of the input data.
    """
    if isinstance(input_data, np.ndarray):
        _validate_image_array(input_data)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=os.getcwd())
        temp_file_path = temp_file.name
        temp_file.close()
        cv2.imwrite(temp_file_path, cv2.cvtColor(input_data, cv2.COLOR_RGB2BGR))
        temp_files.append(temp_file_path)
        return temp_file_path

    return input_data


def _validate_image_array(image_array: np.ndarray) -> None:
    """
    Validate if the provided numpy array is a proper 3D image array.

    Args:
        image_array (np.ndarray): The image array to validate.

    Raises:
        ValueError: If the image array is not valid.
    """
    if image_array.ndim != 3 or image_array.size == 0:
        raise ValueError("Invalid image data. Ensure it's a proper 3D image array.")


def _cleanup_temp_files(temp_files: list) -> None:
    """
    Clean up temporary files.

    Args:
        temp_files (list): A list containing the paths of temporary files to be deleted.
    """
    for temp_file_path in temp_files:
        os.remove(temp_file_path)


def single_input_as_path(func: Callable) -> Callable:
    """
    Decorator to handle a single input as a file path or a numpy array. If the input is a numpy array,
    it saves the image as a temporary file and then uses its path.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """

    @wraps(func)
    def wrapper(input_data: Any, *args, **kwargs) -> Any:
        if isinstance(input_data, np.ndarray):
            _validate_image_array(input_data)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=os.getcwd())
            temp_file_path = temp_file.name
            temp_file.close()
            cv2.imwrite(temp_file_path, cv2.cvtColor(input_data, cv2.COLOR_RGB2BGR))

            try:
                return func(temp_file_path, *args, **kwargs)
            finally:
                os.remove(temp_file_path)
        else:
            return func(input_data, *args, **kwargs)

    return wrapper


@input_as_path
def verify_face_embeddings(embedding1: Any, embedding2: Any, distance_metric: str = 'cosine',
                           threshold: float = 0.45) -> bool:
    """
    Verify if two inputs (either file paths or numpy arrays) represent the same person.
    The decorator 'input_as_path' allows inputs to be either file paths or numpy arrays.

    Args:
        embedding1 (str or np.ndarray): The first input (file path or numpy array).
        embedding2 (str or np.ndarray): The second input (file path or numpy array).
        distance_metric (str): The distance metric to be used for comparison.
        threshold (float): The threshold for deciding if inputs represent the same face.

    Returns:
        bool: True if the distance is below the threshold, indicating the same person.
    """
    result = DeepFace.verify(img1_path=embedding1, img2_path=embedding2, model_name='Facenet512',
                             distance_metric=distance_metric)
    return result['distance'] < threshold


@single_input_as_path
def generate_face_embedding(image_path: Any) -> np.ndarray:
    """
    Generate face embeddings using DeepFace for a given image.
    The decorator 'single_input_as_path' allows the input to be either a file path or a numpy array.

    Args:
        image_path (str or np.ndarray): The file path to the image or a numpy array representing the image.

    Returns:
        np.ndarray: A numpy array representing the face embeddings.
    """
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name='Facenet512', enforce_detection=False)
        return np.array(embedding)
    except ValueError as e:
        raise ValueError(f"Error in generating embedding: {e}")


def verify_embeddings(embedding1: np.ndarray, embedding2: np.ndarray, distance_metric: str = 'cosine',
                      threshold: float = 0.4) -> Tuple[bool, float]:
    """
    Verify if two face embeddings represent the same person based on the specified distance metric.
    This function operates directly on numpy array embeddings.

    Args:
        embedding1 (np.ndarray): The first face embedding.
        embedding2 (np.ndarray): The second face embedding.
        distance_metric (str): The distance metric to be used for comparison ('cosine' or other metrics).
        threshold (float): The threshold for deciding if embeddings represent the same face.

    Returns:
        Tuple[bool, float]: A tuple containing a boolean indicating if the embeddings represent the same face,
                             and the actual distance/similarity score.
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError("Embeddings must have the same shape.")

    norm_embedding1 = embedding1 / np.linalg.norm(embedding1)
    norm_embedding2 = embedding2 / np.linalg.norm(embedding2)

    if distance_metric == 'cosine':
        similarity = np.dot(norm_embedding1, norm_embedding2)
        return similarity > threshold, float(similarity)
    else:
        raise NotImplementedError(f"Distance metric '{distance_metric}' not implemented.")
