from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
