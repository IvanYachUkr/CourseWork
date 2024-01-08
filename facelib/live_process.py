import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def capture_and_detect_faces(camera_id: int = 0, display: bool = True, auto_capture: bool = False) -> Optional[List[np.ndarray]]:
    """
    Captures images from the camera and detects faces using MediaPipe. Can automatically
    capture faces as they are detected.

    Args:
        camera_id: The ID of the camera to use. Default is 0 (default camera).
        display: Whether to display the camera feed with detected faces.
        auto_capture: Automatically capture and return faces as they are detected.

    Returns:
        A list of numpy ndarrays, each representing a processed image of the detected faces,
        or None if no face is detected.
    """
    face_images = []

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise IOError("Cannot open camera")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                     int(bboxC.width * iw), int(bboxC.height * ih)
                        face_img = frame[y:y+h, x:x+w]

                        # Convert the face image from BGR (OpenCV format) to RGB
                        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                        face_images.append(face_img_rgb)

                        if auto_capture:
                            # If auto-capture is enabled, return after first detection
                            return face_images

                if display:
                    cv2.imshow('MediaPipe Face Detection', frame)
                    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    return face_images if face_images else None

import cv2
import numpy as np

def display_extracted_faces(faces: list[np.ndarray], window_name: str = "Extracted Faces", wait_time: int = 0):
    """
    Displays the extracted face images in a single window.

    Args:
    faces: A list of numpy ndarrays, each representing an extracted face.
    window_name: Name of the window in which the images will be displayed.
    wait_time: Time in milliseconds for which each image is displayed.
               The default is 0, which means it waits indefinitely until a key is pressed.

    Raises:
    ValueError: If the faces list is empty.
    """
    if len(faces) == 0:
        raise ValueError("No faces provided for display.")

    for face in faces:
        cv2.imshow(window_name, face)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()




import cv2
import mediapipe as mp
import numpy as np
import time
from typing import List, Optional

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def automated_face_capture(camera_id: int = 0, display: bool = True) -> Optional[np.ndarray]:
    """
    Automatically captures an image from the camera and detects faces using MediaPipe.
    Repeats every 0.5 seconds until a face is detected.

    Args:
        camera_id: The ID of the camera to use. Default is 0 (default camera).
        display: Whether to display the camera feed with detected faces.

    Returns:
        A numpy ndarray representing the processed image of the detected face,
        or None if no face is detected.
    """
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise IOError("Cannot open camera")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image")
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)

                if results.detections:
                    bboxC = results.detections[0].location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    face_img = frame[y:y+h, x:x+w]
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    return face_img_rgb

                if display:
                    cv2.imshow('MediaPipe Face Detection', frame)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

                time.sleep(0.5)  # Wait for 0.5 seconds before next attempt

        finally:
            cap.release()
            cv2.destroyAllWindows()

    return None


from deepface import DeepFace
import cv2
import numpy as np
import time
from typing import Optional

def automated_face_capture_with_deepface(camera_id: int = 0, display: bool = True, max_attempts: int = 5) -> Optional[np.ndarray]:
    """
    Automatically captures an image from the camera and detects faces using DeepFace.
    Retries until a face is detected or the maximum attempts are reached.

    Args:
        camera_id: The ID of the camera to use. Default is 0 (default camera).
        display: Whether to display the camera feed with detected faces.
        max_attempts: Maximum number of attempts to capture a face.

    Returns:
        A numpy ndarray representing the processed image of the detected face,
        or None if no face is detected.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise IOError("Cannot open camera")

    attempts = 0
    while attempts < max_attempts:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue

        try:
            # Use DeepFace for face detection
            detections = DeepFace.detectFace(frame, detector_backend = 'opencv', enforce_detection = False)

            if detections.shape[0] > 0: # If faces are detected
                return detections[0] # Return the first detected face
            else:
                attempts += 1

        except Exception as e:
            print(f"An error occurred in face detection: {e}")
            attempts += 1

        if display:
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

    return None



import cv2
import mediapipe as mp
import numpy as np

def initialize_camera():
    """
    Initialize the camera and return the capture object.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Camera not accessible")
    return cap

def capture_frame(cap):
    """
    Capture a frame from the camera.
    """
    success, frame = cap.read()
    if not success:
        raise Exception("Failed to capture frame")
    return frame

def is_image_blurry(image, threshold=100.0):
    """
    Check if the image is blurry based on the variance of the Laplacian.
    Returns True if the image is blurry, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian < threshold

def is_image_bright_enough(image, brightness_threshold=25.0):
    """
    Check if the image is bright enough.
    Returns True if the average brightness is above the threshold.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray)
    return average_brightness > brightness_threshold


def detect_face(frame, face_detection, blur_threshold=100.0):
    """
    Detect faces in the frame using MediaPipe.
    Returns True if a whole face is detected and the image is not too blurry, False otherwise.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections is None:
        return False

    h, w, _ = frame.shape
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x, y, bbox_width, bbox_height = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height

        xmin = int(x * w)
        ymin = int(y * h)
        box_width = int(bbox_width * w)
        box_height = int(bbox_height * h)

        if xmin > 0 and ymin > 0 and (xmin + box_width) < w and (ymin + box_height) < h:
            # Check for blurriness
            if not is_image_blurry(frame, blur_threshold): # and not is_image_bright_enough(frame):
                return True
    return False


def capture_face_image():
    """
    Capture an image with a face and return it as a numpy array.
    """
    with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        cap = initialize_camera()

        try:
            while True:
                frame = capture_frame(cap)
                if detect_face(frame, face_detection):
                    return np.array(frame)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()


import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


def get_face_landmarks(image):
    # Convert the color space from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect faces
    results = face_mesh.process(image_rgb)

    # Draw face landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                             circle_radius=1),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

    return image


def capture_and_detect():
    # Capture video from the default camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Process the frame and detect face landmarks
        processed_frame = get_face_landmarks(frame)

        # Convert the frame to a NumPy array
        numpy_frame = np.array(processed_frame)

        # You can also display the frame if you want
        # cv2.imshow('MediaPipe FaceMesh', numpy_frame)

        # Hit 'q' on the keyboard to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

        # Return the frame as a NumPy array
        return numpy_frame

    cap.release()

#
# # To run the function and get a frame
# numpy_frame = capture_and_detect()

import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

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

        # Check if a face is detected in the frame
        if face_detected_in_frame(frame):
            # Convert the frame to a NumPy array
            numpy_frame = np.array(frame)

            # Release the capture once the frame is captured
            cap.release()

            # Return the frame as a NumPy array
            return numpy_frame

        # You can add a delay here if needed (e.g., for reducing CPU usage)
        cv2.waitKey(1)

    cap.release()
#
# # To run the function and get a frame
# numpy_frame_with_face = capture_frame_with_face()
#
# def is_full_face_detected(image):
#     # Convert the color space from BGR to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Process the image and detect faces
#     results = face_detection.process(image_rgb)
#
#     # Check if a face is detected with high confidence
#     if results.detections:
#         for detection in results.detections:
#             if detection.score[0] == 1.0:  # Adjust the threshold as needed
#                 # Get the bounding box of the face
#                 bboxC = detection.location_data.relative_bounding_box
#                 height, width, _ = image.shape
#
#                 # Convert the bounding box to pixel coordinates
#                 x, y, w, h = int(bboxC.xmin * width), int(bboxC.ymin * height), \
#                              int(bboxC.width * width), int(bboxC.height * height)
#
#                 # Criteria for a full face (adjust these values as needed)
#                 min_box_width = width * 0.2  # Minimum width of the box to consider it a full face
#                 min_box_height = height * 0.2  # Minimum height of the box to consider it a full face
#
#                 if w > min_box_width and h > min_box_height:
#                     return True
#     return False
#
# def capture_frame_with_face():
#     # Capture video from the default camera
#     cap = cv2.VideoCapture(0)
#
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#             continue
#
#         # Check if a full face is detected in the frame
#         if is_full_face_detected(frame):
#             # Convert the frame to a NumPy array
#             numpy_frame = np.array(frame)
#
#             # Release the capture once the frame is captured
#             cap.release()
#
#             # Return the frame as a NumPy array
#             return numpy_frame
#
#         # You can add a delay here if needed (e.g., for reducing CPU usage)
#         cv2.waitKey(1)
#
#     cap.release()
