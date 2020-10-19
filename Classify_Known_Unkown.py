# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:03:11 2020

@author: jitup
"""
import argparse
import numpy as np
import cv2
from imutils import build_montages
from data_preparation import load_embeddings_img

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-u", "--face_to_compare", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-t", "--tolerance", type=int, default=0.8)
args = vars(ap.parse_args())

# loading embeddings of known face
directory = args["embeddings"]
data = load_embeddings_img(directory)
data = np.array(data)
face_embedding = [d["embedding"] for d in data]

def face_distance(face_embedding, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param face_encodings: List of face embedding to compare
    :param face_to_compare: A face embedding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_embedding) == 0:
        return np.empty((0))

    return np.linalg.norm(face_embedding - face_to_compare, axis=1)
    # return (1 - spatial.distance.cosine(face_embedding, face_to_compare))  # cosine similarity

#loading embeddings of unknow face/face to compare
unknown_embeddings = load_embeddings_img(args["face_to_compare"])
unknown_embeddings = np.array(unknown_embeddings)
tolerance = args["tolerance"]
known = []
unknown = []
#calculating the distance and classifying into known-unknown
for unknown_embedding in unknown_embeddings:
        distances = face_distance(face_embedding, unknown_embedding["embedding"])
        result = list(distances <= tolerance)
        face = cv2.imread(unknown_embedding["imagePath"])
        face = cv2.resize(face, (96, 96))
        if True in result:
            known.append(face)
        else:
            unknown.append(face)

montageK = build_montages(known, (96, 96), (5, 5))[0]
# show the output montage for known person
title = "Known faces"
cv2.imshow(title, montageK)
cv2.waitKey(0)

montageUNK = build_montages(unknown, (96, 96), (5, 5))[0]
# show the output montage for unknown person
title = "UNKnown faces"
cv2.imshow(title, montageUNK)
cv2.waitKey(0)