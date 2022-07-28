
from scipy.spatial import distance
import cv2
import numpy as np


# This function calculates and return eye aspect ratio
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#
#     ear = (A + B) / (2 * C)
#     return ear
