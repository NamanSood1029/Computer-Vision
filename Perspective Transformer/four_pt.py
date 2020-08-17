import numpy as np
import cv2
import argparse


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


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coords", help="comma seperated list of source points")
args = vars(ap.parse_args())
# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them
image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype="float32")
# apply the four point tranform to obtain a "birds eye view" of
# the image
warped = four_point_transform(image, pts)
# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
