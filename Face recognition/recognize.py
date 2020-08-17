# import necessary packages
import numpy as np
import imutils
import pickle
import os
import cv2
import argparse

# Now our main objective after learning all the weights and models is to use it on actual images
# We will undergo the same process as we did in image_recog.py but here we have new input
# We will again find the face of the image, get a bounding box, get an embedding matrix and pass it through
# The SVM will predict the labels

# Argument parser arguments
# image - path to input image
# detector - path to deep learning face detector -> learn ROI
# embedding model - extract 128-D vector from ROI
# recognizer - SVM recognizer used
# le - path to our label encoder
# confidence - optional threshold

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-d", "--detector", required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True, help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True, help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True, help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face detector...")
# detector - pre-trained caffe DL model to detect location of faces (identification)
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
# pre-trained DL model to calculate the Face embeddings
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
# Our linear SVM recognition model
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
# Our label encoder
le = pickle.loads(open(args["le"], "rb").read())

# Load image
image = cv2.imread(args['image'])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# construct a blob from the image - preprocessing the image
imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                  (104.0, 177.0, 123.0), swapRB=False, crop=False)
# Pass it through the image detection algorithm
detector.setInput(imageBlob)
detections = detector.forward()

# Face detection and bounding boxes
# looping over detections

for i in range(0, detections.shape[2]):
    # extract confidence
    confidence = detections[0, 0, i, 2]
    # filter out weak ones
    if confidence > args['confidence']:
        # compute (x,y) coordinates of the box for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        if fW < 20 or fH < 20:
            continue
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        # classification
        pred = recognizer.predict_proba(vec)[0]
        j = np.argmax(pred)
        proba = pred[j]
        name = le.classes_[j]

        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow('Image', image)
cv2.waitKey(0)