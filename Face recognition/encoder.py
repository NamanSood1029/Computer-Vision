from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# dlib will be the pre-trained network to construct the 128-d embeddings for each face

# dataset -> path to our dataset containing the images
# encodings -> face encodings written to the file it points to
# detection-method -> which method to use (for DETECTING faces), CNN or hog

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

print('[INFO] quantifying faces...')
imagePaths = list(paths.list_images(args['dataset']))

# initialize the encodings and labels list
knownEncodings = []
knownNames = []

# looping over image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the name of the folder -> label name (which contains the images)
    print('[INFO] processing image {}/{}'.format(i+1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load image and convert from BGR (openCV default) to dlib (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detecting the (x,y) values of the bounding boxes for each face of input image
    # passing in two arguments - image and the model to use (cnn or hog)
    boxes = face_recognition.face_locations(rgb, model=args['detection_method'])

    # converting them to encodings
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        # add each name and their respective encoding to the set
        knownEncodings.append(encoding)
        knownNames.append(name)

print('[INFO] serializing encodings')
data = {'encodings':knownEncodings,'names':knownNames}
f = open(args['encodings'], 'wb')
f.write(pickle.dumps(data))
f.close()

