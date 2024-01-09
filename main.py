from sklearn.datasets import fetch_lfw_people
#
#
# lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=1)
# #
import matplotlib.pyplot as plt
# #
# # # Display the first image in the dataset
# # plt.imshow(lfw_people.images[1], cmap='gray')
# # plt.title(f"Label: {lfw_people.target_names[lfw_people.target[0]]}")
# # plt.show()
# #
#
import cv2
import mediapipe as mp
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
#
from facelib.preprocessing import *
from facelib.extract import *
#
# # Select an example image from the dataset
# example_image = lfw_people.images[3]  # First image in the dataset
#
# # Preprocess the example image
# preprocessed_image = convert_to_rgb(example_image)
#
# # Initialize MediaPipe Face Detection.
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
# # Detect faces in the preprocessed image.
# detected_image = detect_faces(preprocessed_image)
#
# # Display the detected image.
# plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR))
# plt.axis("off")
# plt.show()
#
#
# main_image = convert_to_rgb(example_image)
#
# face_borders = detect_faces_borders(main_image)
#
# faces = crop_faces(main_image, face_borders)
#
# display_cropped_faces(faces)

#from facelib.live_process import capture_frame_with_face,capture_and_detect, capture_face_image, capture_and_detect_faces, display_extracted_faces, automated_face_capture, automated_face_capture_with_deepface
from facelib.detect_live import capture_frame_with_face
from facelib.emotion_detection import detect_emotions_in_frame
# Example usage
# capture_and_detect_faces()
#
# detected_faces = capture_and_detect_faces(display=True)
#
# print(detected_faces)
#
# display_extracted_faces(detected_faces)

# detected_face = automated_face_capture()
# if detected_face is not None:
#     display_cropped_faces([detected_face])

# detected_face = automated_face_capture_with_deepface()
# if detected_face is not None:
#     display_cropped_faces([detected_face])



# detected_face = capture_face_image()
# if detected_face is not None:
#     display_cropped_faces([detected_face])


# detected_face = capture_frame_with_face()
# if detected_face is not None:
#     display_cropped_faces([detected_face])

# main_image =  convert_to_rgb(detected_face)
#
# face_borders = detect_faces_borders(main_image)
#
# faces = crop_faces(main_image, face_borders)
#
# display_cropped_faces(faces)

# print(np.array(faces[0]))
# emotions = detect_emotions_in_frame(faces[0])
# print("Detected Emotions:", emotions)

# from facelib.embeddings import generate_face_embedding, verify_faces
# #import facelib.embeddings as em
# embedding1 = generate_face_embedding('Figure_3.png')
# embedding2 = generate_face_embedding('Figure_2.png')
#
# similarity = verify_faces(embedding1, embedding2)
# print(f"Cosine Similarity: {similarity}")
#
#
from facelib.embeddings import face_embedding_from_path, verify_faces
embedding1 = face_embedding_from_path('Figure_1.png')
embedding2 = face_embedding_from_path('Figure_3.png')
print("Are the faces the same?", verify_faces(embedding1, embedding2))