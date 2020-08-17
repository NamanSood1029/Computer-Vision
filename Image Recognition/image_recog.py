# Image face recognition

# Main algorithm at work -> FaceNet
# Working of openCV face recognition
# We have an input image -> detect a face -> transform it and crop it
# We then proceed to pass it through a neural network which gives us a 128-node embedding layer at the end
# This is used as a feature-set for the image
# We can use this embedding layer along with a ML model -> SVM, Randomforest etc. to predict our image labels

# The loss function we use is the triplet loss function
# Each input batch includes three images -> anchor, positive image, negative image
# Anchor is our current face with identity A, the second image is our positive image - contains face of A
# The negative image - does not have the same identity and could belong to person B, C or Y
# We compute the 128-d embeddings and then through the triplet loss function -> proceed to update the weights

# Main directories
# Dataset/ - face images in sub folders by name
# images/ - contains test images used to verify operations of the model
# Face_detection_model/ - pre-trained caffe DL model to detected faces
# output/ - output pickle files, which include:
    # embeddings.pickle - serialized face embeddings file -> embeddings for each face stored here
    # le.pickle - our label encoder, contains labels of people the model can recognize
    # recognizer.pickle - our classifier, here we are using an SVM

# extract_embeddings.py - responsible for ggenerating the 128-D vector describing a face.
# Torch deep learning model which produces the 128-D facial embeddings
# A training model which will train our script
# recognize which will recognize faces in the image
# video recognizer for real-time recognition

# STEP 1 EXTRACT EMBEDDINGS

from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Dataset - input dataset of face images
# embeddings - output embeddings file which will be computed and serialized
# detector - caffe-based deep learning face detector for face DETECTION
# embedding model - extract 128-D facial embedding vector
# confidence - optional filtering for weak face detections

ap = argparse.ArgumentParser
ap.add_argument('-i','--dataset', required=True, help='path to input directory of faces+images')
ap.add_argument('-e', '--embeddings',required=True,help='path to serialized output database of embeddings')
ap.add_argument('-d','--detector',required=True,help="Path to opencv's deep learning face detector")
ap.add_argument('-m', '--embedding-model', required=True, help='openCV deep learning embedding model')
ap.add_argument('-c','--confidence',type=float, default=0.5, help='Minimum probability to filter weak detections')
args = vars(ap.parse_args())

# We input the folders, so we still need to locate the files for the detector and model

print('[INFO] loading the face detector...')
protoPath = os.path.sep.join([args['detector'],'deploy.prototxt'])
modelPath = os.path.sep.join([args['detector'],'res10_300x300_ssd_iter_140000.caffemodel'])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print('[INFO] loading face recognizer...')
embedder = cv2.dnn.readNetFromTorch(args['embedding_model'])

print('[INFO] quantifying faces...')
imagePaths = list(paths.list_images(args['dataset']))

# Initialize lists of embeddings and people's names
knownEmbeddings = []
knownNames = []

# Initialize the total number of faces processed
total = 0

# loop over image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the name of the person from the image path
    print('[INFO] processing image {}/{}'.format(i+1,len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load image, resize it (maintaining aspect ratio) and grab image dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

# Example
# from imutils import paths
# import os
# imagePaths = list(paths.list_images("dataset"))
# imagePath = imagePaths[0]
# imagePath
# output - 'dataset/adrian/00004.jpg'
# imagePath.split(os.path.sep)
# output -['dataset', 'adrian', '00004.jpg']
# imagePath.split(os.path.sep)[-2]
# output - 'adrian'

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0,177.0,123.0), swapRB=False, crop=False)
    # apply the face detector to localize all faces
    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) > 0:
        # assuming that each image has only one face -> find bounding box with image of largest probability
        # We assume each image has one face as it is training data and each dataset should have images
        # corresponding to that person only.
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        # ensure detection with largest probability also meets our criteria
        if confidence > args['confidence']:
            # compute the coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            # extract region of interest of the face and its dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure sufficiently large
            if fW < 20 or fH < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
            # We pass it through the embedder to get our 128-D vector
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # add the name of the person + face embedding to the list
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

        print("[INFO] serializing {} encodings...".format(total))
        data = {'embeddings': knownEmbeddings, 'names': knownNames}
        f = open(args['embeddings'],'wb')
        f.write(pickle.dumps(data))
        f.close()
