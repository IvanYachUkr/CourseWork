import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.9)

def is_image_blurry(image, threshold=80.0):
    """
    Check if the image is blurry based on the variance of the Laplacian.
    Returns True if the image is blurry, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian < threshold

def face_detected_in_frame(image):
    # Convert the color space from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect faces
    results = face_detection.process(image_rgb)

    # Check if a face is detected with high confidence
    if results.detections:
        for detection in results.detections:
            if detection.score[0] > 0.9:  # Adjust the threshold as needed
                return True
    return False

def capture_frame_with_face():
    # Capture video from the default camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Check if a face is detected in the frame and the frame is not blurry
        if face_detected_in_frame(frame) and not is_image_blurry(frame):
            # Convert the frame to a NumPy array
            numpy_frame = np.array(frame)

            # Release the capture once the frame is captured
            cap.release()

            # Return the frame as a NumPy array
            return numpy_frame

        # You can add a delay here if needed (e.g., for reducing CPU usage)
        cv2.waitKey(1)

    cap.release()
