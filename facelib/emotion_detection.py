# emotion_detection.py
import os
import tempfile

import cv2
from deepface import DeepFace


def detect_emotions_in_image(image_path):
    """
    Detects emotions in an image file.

    Args:
    image_path (str): Path to the image file.

    Returns:
    dict: A dictionary containing the detected emotions and their probabilities.
    """
    try:
        image = cv2.imread(image_path)
        analysis = DeepFace.analyze(image, actions=('emotion',), enforce_detection=False)
        return analysis['emotion']
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return None



def detect_emotions_in_frame(frame):
    try:
        # Check if the frame is None
        if frame is None:
            raise ValueError("Received a None frame")

        # Convert the color space from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save the frame to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            file_name = tmpfile.name
            cv2.imwrite(file_name, frame_rgb)

        # Analyze the emotions in the frame
        analysis = DeepFace.analyze(file_name, actions=['emotion'], enforce_detection=False)

        # Delete the temporary file
        os.remove(file_name)

        # Check the structure of the analysis result and extract emotions
        if isinstance(analysis, dict) and 'emotion' in analysis:
            emotions = analysis['emotion']
        elif isinstance(analysis, list) and len(analysis) > 0:
            emotions = analysis[0]['emotion']
        else:
            print("Unexpected format of analysis results")
            return None

        # Sort the emotions by their values and get the top 3
        dominant_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)[:3]
        return dominant_emotions

    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return None