import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional

def crop_faces(image: np.ndarray, face_borders: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """
    Crops faces from the given image based on the provided face borders.

    Args:
    image: A numpy ndarray representing the original image.
    face_borders: A list of tuples, each representing the border of a detected face.
                  Each tuple contains (x, y, width, height).

    Returns:
    A list of numpy ndarrays, each representing a cropped face image.
    """
    cropped_faces = []

    for border in face_borders:
        if len(border) == 4:
            x, y, w, h = border
            # Ensure coordinates are within image bounds
            x, y = max(0, x), max(0, y)
            w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)

            # Crop the face from the image
            face_image = image[y:y+h, x:x+w]
            cropped_faces.append(face_image)
        else:
            raise ValueError("Face border does not have exactly four elements.")

    return cropped_faces



def display_cropped_faces(cropped_faces: list):
    """
    Displays cropped faces using Matplotlib.

    Args:
    cropped_faces: A list of numpy ndarrays, each representing a cropped face image.
    """
    if not cropped_faces:
        print("No faces to display.")
        return

    # Create a subplot for each cropped face
    num_faces = len(cropped_faces)
    fig, axes = plt.subplots(1, num_faces, figsize=(5 * num_faces, 5))

    if num_faces == 1:
        # Only one face, no need for a loop
        axes.imshow(cropped_faces[0])
        axes.axis('off')
    else:
        for ax, face in zip(axes, cropped_faces):
            ax.imshow(face)
            ax.axis('off')

    plt.show()

# Example usage (assuming `faces` is a list of cropped face images)
# display_cropped_faces(faces)

