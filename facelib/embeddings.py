# from deepface import DeepFace
#
#
#
#
# def generate_face_embedding(image_path):
#     try:
#         embedding = DeepFace.represent(img_path=image_path, model_name='Facenet')
#         return embedding
#     except Exception as e:
#         print(f"Error in generating embedding: {e}")
#         return None
#
#
# def verify_faces(embedding1, embedding2, distance_metric='cosine', threshold=0.4):
#     from deepface.commons import distance as dst
#     distance = dst.findCosineDistance(embedding1, embedding2)
#     if distance <= threshold:
#         return "Faces are similar", distance
#     else:
#         return "Faces are not similar", distance
#
#
# import mediapipe as mp
# import cv2
# import numpy as np
# #import ImageEmbedder
# from mediapipe.tasks.python import vision
# mp_embedder = vision.image_embedder
#
#
# def generate_image_embedding(image_path):
#     with mp_embedder.ImageEmbedder(model_path=mp_embedder.SELFIE_V2_EMBEDDER_MODEL_PATH) as embedder:
#         image = cv2.imread(image_path)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         result = embedder.embed(image_rgb)
#         if result.embeddings:
#             return result.embeddings[0].feature_vector
#         else:
#             return None
#
# def compute_similarity(embedding1, embedding2):
#     if embedding1 is None or embedding2 is None:
#         return 0  # Return 0 similarity if either embedding is None
#
#     similarity = mp_embedder.ImageEmbedder.cosine_similarity(embedding1, embedding2)
#     return similarity
#


from mediapipe.tasks import python
from mediapipe.tasks.python import vision




def initialize_image_embedder(model_path='embedder.tflite', l2_normalize=True, quantize=True):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)
    embedder = vision.ImageEmbedder.create_from_options(options)
    return embedder

embedder = initialize_image_embedder()

import mediapipe as mp

def generate_image_embedding(embedder, image_path):
    image = mp.Image.create_from_file(image_path)
    embedding_result = embedder.embed(image)
    if embedding_result.embeddings:
        return embedding_result.embeddings[0]
    else:
        return None


def calculate_cosine_similarity(embedding1, embedding2):
    similarity = vision.ImageEmbedder.cosine_similarity(embedding1, embedding2)
    return similarity
