# bubble scanner - used for grading in OMR sheets in exams like SATs
# 7 steps to build a scanner+grader
# 1 - detect the grade sheet in the image
# 2 - extract the top-down view of that image (birds-eye-view)
# 3 - extract the set of bubbles from the transformed sheet
# 4 - sort the bubbles and questions into rows
# 5 - Determine marked answers for each row
# 6 - Compare the correct answer in our answer key to determine if it was correct
# 7 - Repeat the steps for all questions

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the input image')
args = vars(ap.parse_args())

# define the answer key
key = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# These will give the silhouette of the image, after Canny edges/outline detection
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# Find lines that correspond to the exam sheet itself - next objective
# We will first find all contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# ensure that there were contours found
if len(cnts) > 0:
    # We sort the contours based on their area to find the contour
    # corresponding to the perimeter/boundary of the exam sheet
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        # We approximate the contour and its shape
        # Here, we check the number of vertices of the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            docCnt = approx
            break

paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# we proceed to binarize the input (our input now is the top-down view)
# This will result in a black background and a white foreground
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# We will again apply contour extraction process on this view
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
    # Compute bounding box of contour and derive aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # in order to label the contours, the region should be sufficiently spaced
    # It should be sufficiently tall as well, with aspect ratio approximately 1
    # This will give us the contours around each and every circle
    if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
        questionCnts.append(c)

# Grading method
# Sort the question contours top-to-bottom and initialize total correct answers
# Ensure that rows that are closer to the top of the exam - appear first in sorted list
questionCnts = contours.sort_contours(questionCnts, method='top-to-bottom')[0]
correct = 0

# Each has 5 possible answers, loop over in batches of 5
# Apply numpy array slicing and contour sorting to sort from left to right
# As we have sorted from top to bottom, we know the 5 bubbles for each q will appear sequentially
# We do not know whether the bubbles will eb sorted from left to right
# The inner sort_contours takes care of it
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # sort the contours from left to right and initialize index of bubbled
    cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None

    # loop over sorted contours from left to right
    for (j, c) in enumerate(cnts):
        # construct mask to reveal only current bubble for the question
        mask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(mask, [c], -1, 255, -1)

        # apply mask to the threshold image, count number of non-zero pixels
        # in the buble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        # As the shaded region is white (after applying the binarize)
        # The bubble with the most non-zero pixels is shaded
        # It is the index of the q i.e. the option number
        # keep comparing with bubbled[0] to replace with max
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    color = (0, 0, 255)
    k = key[q]

    if k == bubbled[1]:
        color = [0, 255, 0]
        correct += 1

    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

score = (correct / 5.0) * 100
print('[INFO] score: {:.2f}%'.format(score))
cv2.putText(paper, '{:.2f}%'.format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 0, 255), 2)

cv2.imshow('Original', image)
cv2.imshow('Exam', paper)
cv2.waitKey(0)
