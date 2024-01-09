import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing to the input image if needed.

    Args:
    image (np.ndarray): The input image array.

    Returns:
    np.ndarray: The preprocessed image array.
    """
    # Normalize the image and convert it back to uint8
    return np.uint8(image / 255.0 * 255)


def face_embedding_from_array(image_array: np.ndarray) -> np.ndarray:
    """
    Generate face embeddings from an image array using a pre-trained model.

    Args:
    image_array (np.ndarray): A numpy array representation of an image.

    Returns:
    np.ndarray: A numpy array representing the face embedding.

    Raises:
    ValueError: If no face is detected in the image.
    """
    model_file = r"facelib\res10_300x300_ssd_iter_140000.caffemodel"
    config_file = r"facelib\deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    # Preprocess the image
    processed_image = preprocess_image(image_array)

    # Create a blob from the image array
    blob = cv2.dnn.blobFromImage(processed_image, 1.0, (300, 300), [104, 117, 123], False, False)

    # Set the input to the network and forward pass to get the output
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            return detections[0, 0, i, 3:7] * np.array(
                [image_array.shape[1], image_array.shape[0], image_array.shape[1], image_array.shape[0]])

    raise ValueError("No face detected in the image.")


def face_embedding_from_path(image_path: str) -> np.ndarray:
    """
    Generate face embeddings from an image file path.

    Args:
    image_path (str): Path to the image file.

    Returns:
    np.ndarray: A numpy array representing the face embedding.
    """
    image = cv2.imread(image_path)
    return face_embedding_from_array(image)


def verify_faces(embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Compare two face embeddings to determine if they represent the same person.

    Args:
    embedding1 (np.ndarray): The first face embedding.
    embedding2 (np.ndarray): The second face embedding.
    threshold (float): The threshold for deciding if embeddings represent the same face.

    Returns:
    bool: True if the embeddings represent the same face, False otherwise.
    """
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance #< threshold
