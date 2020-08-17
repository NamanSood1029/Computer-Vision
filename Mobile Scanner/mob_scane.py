from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

def order_points(pts):
    # We initialize a list of coordiantes in order top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype='float32')

    # To recognize which points are top-left and so on
    # The top-left point will have the smallest sum while the bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute difference between the points for the other two points
    # The top-right point will have the smallest difference
    # Bottom-left will have the largest difference

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # unpack the points we have
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute width of new image - maximum((bottom-right & bottom-left),(top-right & top-left))
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute height of new image - maximum( (top-right and bottom-right),(top-left and bottom-left))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # After dimensions of new image -> destination points will be specified to obtain a birds eye view
    # First argument - top left of the image -> 0,0 in our final result
    # Second argument - top right corner
    # Third argument - bottom right corner
    # Fourth argument - bottom left corner
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype='float32')

    # Two arguments -> rect (list of 4 ROI points in the original) and dst (list of transformed points)
    # ROI - Region of interest
    # We get a transformation matrix which we then apply using the warp perspective function
    # We pass in our image, transform matrix M along with width and height of the final image.
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image to be scanned')
args = vars(ap.parse_args())

# STEP-1 EDGE DETECTION

# Load image and compute ratio of old height to new height, aspect ratio values, clone and resize it.
# To make the edge detection faster, we will resize the image to have a height of 500 pixels
image = cv2.imread(args['image'])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

print('STEP 1, Edge detection')
cv2.imshow('Image', image)
cv2.imshow('Edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# STEP-2 FINDING CONTOURS

# We will assume the largest rectangle/ polygon to be our actual image sheet.
# Finding the contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# Sorting and keeping only the largest contours to speed up computatoin
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# Loop over the contours
for c in cnts:
    # approximate the contour
    # We first get the contour length, if passed true, it means that the contour is closed and vice versa
    peri = cv2.arcLength(c, True)
    # Approximates a contour shape to another shape with less number of vertices depending on our precision
    # If we want to find square in an image but we don't get a perfect square, we can use this function.
    # Second parameter - epsilon - maximum distance from contour to approximate contour
    # A wise selection is needed to get the correct output, returns number of points
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # If our approx. contour has 4 points, we can assume we've found our screen
    if (len(approx)) == 4:
        screenCnt = approx
        break

print('STEP 2, Find contours of paper')
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow('Outline', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

warped = four_point_transform(orig, screenCnt.reshape(4, 2)*ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = 'gaussian')
warped = (warped >T).astype('uint8')*255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)