# Face detection algorithm

# A dnn module is already present in the official release of openCV which can be found in face_detector
# It is a caffe based face detector - caffe is a deep learning framework.
# Caffe allows us to build our own DNN or access pre-made nets.
# Originally designed for machine-vision tasks (CNNs), speech & text, RNNs and other supports were added later.

# We require two set of files
# .prototxt file -> defines model architectures and their layers
# .caffemodel file -> contains the weights of the actual layers
# Both are required to train the model using Caffe for deep learning

# The face detector algorithm is based on Single shot detectors (SSD) framework with a ResNet base network

# Single shot detection: Main example -> YOLO
# Input data is an image and we have to detect pedestrians and cars. Number of objects are unknown.
# Approach taken is -> Fixed number of bounding boxes and classifications
# We classify each bounding box as an object or not an object.
# If we ignore all boxes that are not objects -> We produce the required number of boxes.
# We separate input into grid cells
# We generate a y vector for probability of object is present, which object it is along with the bounding box params.
# x and y are relative to the center of the grid cell.
# The cell with the center of the object is associated as the center.
# For multiple objects in a grid cell, we use multiple anchor boxes


import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Loading the model
print('[INFO] loading model...')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

image = cv2.imread(args['image'])
(h, w) = image.shape[:2]

# A form of image pre-processing
# Arguments
# Image - the image we want to pre-process before passing it through the deep neural network
# Scale-factor - After performing mean subtraction, scale image by some factors.
# It is important to note scale-factor should be 1/sigma as we are multiplying the input channels after mean subtraction
# by the scale-factor
# Size -  We supply the spatial size that the neural network expects. For most it is 224x224, 227x227 etc.
# The mean value that we want to be subtracted from the image of every channel
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

print('[INFO] computing object detections..')
net.setInput(blob)
detections = net.forward()

# Network produces output blob with a shape 1x1xNx7 where N is a number of
# detections and an every detection is a vector of values
# [batchId, classId, confidence, left, top, right, bottom]

# LOOP over the detections
for i in range(0, detections.shape[2]):
    # Extract the confidence/probability associated with the prediction
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections by ensuring it's greater than the minimum confidence
    if confidence > args['confidence']:
        # compute (x,y) coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        # Draw bounding box of face along with associated probability
        text = "{:.2f}".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow('Output', image)
cv2.waitKey(0)
