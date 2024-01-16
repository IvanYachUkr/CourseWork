from emotion_detection import detect_emotions_in_image, extract_dominant_emotions
from preprocessing import adjust_brightness_contrast, apply_gaussian_blur, normalize_image, equalize_histogram
from preprocessing import load_image, convert_to_rgb
from detect_live import capture_frame_with_face
import cv2
import detect_live
from detect_face import detect_faces, detect_faces_borders, crop_faces
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from face_recognition import verify_face_embeddings


def display_image_with_emotions(image_path, emotions):
    """
    Displays an image and a bar graph of the detected emotions next to it.

    Args:
    image_path (str): Path to the image file.
    emotions (dict): Dictionary containing detected emotions and their probabilities.
    """
    # Load and convert the image
    image = load_image(image_path)
    image_rgb = convert_to_rgb(image)

    # Create subplot for image and bar graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image
    ax1.imshow(image_rgb)
    ax1.axis('off')
    ax1.set_title('Detected Face')

    # Extract data for the bar graph
    emotion_names = list(emotions.keys())
    emotion_values = list(emotions.values())

    # Create a bar graph for emotions
    ax2.bar(emotion_names, emotion_values, color='skyblue')
    ax2.set_title('Emotion Probabilities')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 100)

    # Show the plot
    plt.tight_layout()
    plt.show()


def test_emotion_detection_from_lfw():
    print("Testing emotion detection on an image from the LFW dataset")
    try:
        emotions = detect_emotions_in_image('img.jpg')
        if emotions:
            dominant_emotions = extract_dominant_emotions(emotions)
            print("Detected 3 dominant emotions:", dominant_emotions)
            display_image_with_emotions('img.jpg', emotions)
        else:
            print("No emotions detected.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    test_emotion_detection_from_lfw()


def display_original_and_processed_images(original, processed, title):
    """
    Displays the original and processed images side by side.

    Args:
    original (np.ndarray): The original image.
    processed (np.ndarray): The processed image.
    title (str): The title for the subplot of the processed image.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    ax1.imshow(original)
    ax1.axis('off')
    ax1.set_title('Original Image')

    # Display the processed image
    ax2.imshow(processed)
    ax2.axis('off')
    ax2.set_title(title)

    plt.tight_layout()
    plt.show()


def test_preprocessing_functions(image_path):
    print("Testing preprocessing functions on an image")

    # Load and convert the image
    original_image = load_image(image_path)
    image_rgb = convert_to_rgb(original_image)

    # Apply preprocessing functions
    brightness_contrast_adjusted = adjust_brightness_contrast(image_rgb, brightness=30, contrast=30)
    gaussian_blurred = apply_gaussian_blur(image_rgb, (29, 29))
    normalized_image = normalize_image(image_rgb)
    histogram_equalized = equalize_histogram(image_rgb)

    # Display results
    display_original_and_processed_images(image_rgb, brightness_contrast_adjusted, 'Brightness & Contrast Adjusted')
    display_original_and_processed_images(image_rgb, gaussian_blurred, 'Strong Gaussian Blur Applied')
    display_original_and_processed_images(image_rgb, normalized_image, 'Normalized Image')
    display_original_and_processed_images(image_rgb, histogram_equalized, 'Histogram Equalized')


if __name__ == "__main__":
    test_preprocessing_functions('img.jpg')


def display_captured_frame():
    frame = capture_frame_with_face()
    if frame is not None:
        cv2.imshow('Captured Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No suitable frame captured.")


if __name__ == "__main__":
    display_captured_frame()


def add_white_strip(image, strip_height=60):
    """
    Adds a white strip to the top of the image.

    Args:
    image (np.ndarray): The original image.
    strip_height (int): Height of the white strip to add.

    Returns:
    np.ndarray: The image with a white strip added to the top.
    """
    white_strip = np.full((strip_height, image.shape[1], 3), 255, dtype=np.uint8)
    return np.vstack((white_strip, image))


def annotate_image(image, kernel_size, is_blurry):
    """
    Annotates the image with kernel size text and adds a colored border.
    """
    if kernel_size:
        # Increase font size and thickness
        font_scale = 3.5  # Larger font size
        thickness = 7  # Thicker text for better visibility

        # Calculate position for text to be in the top left corner
        text_x = 10
        text_y = int(image.shape[0] * 0.2)

        # Set text color to black (BGR: 0,0,0)
        text_color = (0, 0, 0)

        cv2.putText(image, f"{kernel_size[0]}x{kernel_size[1]}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    border_color = (0, 0, 255) if is_blurry else (0, 255, 0)  # Red if blurry, green if not
    bordered_image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)

    return bordered_image


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize the image to a specified width or height while maintaining aspect ratio.
    """

    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def create_composite_image(images, display_width=400):
    """
    Creates a composite image by arranging a list of images in a grid.
    """
    resized_images = [resize_with_aspect_ratio(img, width=display_width // 2) for img in images]
    row1 = np.hstack(resized_images[:2])  # First row with 2 images
    row2 = np.hstack(resized_images[2:])  # Second row with 2 images
    return np.vstack((row1, row2))  # Stack the two rows


def test_blur_effect_on_face_detection(image_path, display_width=400):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    blur_sizes = [None, (3, 3), (5, 5), (11, 11)]
    images = [image]  # Start with the original image

    # Apply different levels of Gaussian blur
    for blur_size in blur_sizes[1:]:
        blurred_image = cv2.GaussianBlur(image, blur_size, 0)
        images.append(blurred_image)

    # Annotate and resize images for the composite
    annotated_images = []
    for index, img in enumerate(images):
        is_blurry = detect_live.is_image_blurry(img) if index != 0 else False
        kernel_size = blur_sizes[index]
        annotated_image = annotate_image(img.copy(), kernel_size, is_blurry)
        resized_annotated_image = resize_with_aspect_ratio(annotated_image, width=display_width // 2)
        annotated_images.append(resized_annotated_image)

    # Create composite image
    composite_image = create_composite_image(annotated_images, display_width)

    # Display the composite image
    cv2.imshow('Blur Effect on Face Detection', composite_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_blur_effect_on_face_detection('img.jpg')


def test_detect_and_crop_faces(image_path: str) -> None:
    """
    Test function to demonstrate face detection and cropping.

    Args:
    image_path (str): The file path of the image to process.
    """
    # Load image
    image = np.array(Image.open(image_path))

    # Detect faces and get borders
    annotated_image = detect_faces(np.copy(image))
    face_borders = detect_faces_borders(image)

    # Crop faces
    cropped_faces = crop_faces(image, face_borders)

    # Determine the number of subplots required
    num_faces = len(cropped_faces)
    total_plots = 1 + num_faces  # One for the original image and one for each cropped face

    # Create figure with appropriate number of subplots
    fig, axes = plt.subplots(1, total_plots, figsize=(5 * total_plots, 5))

    # Display original image with detected faces
    axes[0].imshow(annotated_image)
    axes[0].set_title("Detected Faces")
    axes[0].axis('off')

    # Display each cropped face
    for i, face in enumerate(cropped_faces, start=1):
        axes[i].imshow(face)
        axes[i].set_title(f"Cropped Face {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == '__main__':
    test_detect_and_crop_faces('img.png')


def test_face_recognition(image_path1: str, image_path2: str) -> None:
    """
    Test function to demonstrate face recognition.

    Args:
    image_path1 (str): The file path of the first image to process.
    image_path2 (str): The file path of the second image to process.
    """
    # Load images
    image1 = np.array(Image.open(image_path1))
    image2 = np.array(Image.open(image_path2))

    # Verify if the embeddings represent the same person
    same_person = verify_face_embeddings(image1, image2)

    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display first image
    ax1.imshow(image1)
    ax1.set_title("Image 1")
    ax1.axis('off')

    # Display second image
    ax2.imshow(image2)
    ax2.set_title("Image 2")
    ax2.axis('off')

    # Display result as a main title
    plt.suptitle('Same Person: {}'.format('Yes' if same_person else 'No'),
                 fontsize=16, color='green' if same_person else 'red')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the main title
    plt.show()


if __name__ == '__main__':
    test_face_recognition('img.png', 'img.jpg')
