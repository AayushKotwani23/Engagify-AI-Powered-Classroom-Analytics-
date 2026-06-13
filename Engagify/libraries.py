import cv2
import mediapipe as mp
import tensorflow as tf
from deepface import DeepFace

print("OpenCV Version:", cv2.__version__)
print("TensorFlow Version:", tf.__version__)
print("DeepFace Working:", DeepFace.build_model("VGG-Face"))
