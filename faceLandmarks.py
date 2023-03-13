# import required libraries

from imutils import face_utils

import numpy as np

import argparse

import imutils

import dlib

import cv2

ap = argparse.ArgumentParser()

# –-image is a path to the input image

ap.add_argument(“-p”, “–shape-predictor”, required=True,

help=”path to facial landmark predictor”)

ap.add_argument(“-i”, “–image”, required=True,

help=”path to input image”)

args = vars(ap.parse_args())

# initialize built-in face detector in dlib

detector = dlib.get_frontal_face_detector()

# initialize face landmark predictor

predictor = dlib.shape_predictor(args[“shape_predictor”])

# load input image, resize it, and convert it to grayscale

image = cv2.imread(args[“image”])

image = imutils.resize(image, width=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image

rects = detector(gray, 1)

# # # End of converting to grayscale.

for (i, rect) in enumerate(rects):

# predict facial landmarks in image and convert to NumPy array

shape = predictor(gray, rect)

shape = face_utils.shape_to_np(shape)

# convert to OpenCV-style bounding box

cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the face number and draw facial landmarks on the image

cv2.putText(image, “Face #{}”.format(i + 1), (x – 10, y – 10),

cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

for (x, y) in shape:

cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# show the resulting output image

cv2.imshow(“Output”, image)

cv2.waitKey(0)
