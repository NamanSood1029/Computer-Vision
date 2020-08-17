from __future__ import print_function
import numpy as np
import argparse
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# We use a very big filter, to blur out image strongly so that the edges are more prominent

blurred = cv2.GaussianBlur(gray, (11, 11), 0)
cv2.imshow("Image", image)

# Apply canny edge detection

# All gradient values below 30 are considered non-edges while any above 150 are considered sure edges
edged = cv2.Canny(blurred, 30, 150)
cv2.imshow("Edges", edged)

# Tuple values - (1) image after applying contour detection - edged image, function is destructive to image
# that we pass in, better make backup before applying the function.
# (2) - Contours themselves (cnts)
# (3) - The heirarchy of contours
# We use cv2.RETR_EXTERNAL to retrieve only the outermost contours (i.e. along the boundary).
# We can also pass in cv2.RETR_LIST to grab all contours.
# Other methods - using cv2.RETR_COMP and cv2.RETR_TREE
# Last argument - how to approximate contours, we use CHAIN_APPROX_SIMPLE to compress horizontal, vertical and
# diagonal segments into their end-points only. This saves computation and memory but retrieving all points
# is mostly unnecessary and wasteful

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("I count {} coins in this image".format(len(cnts)))
coins = image.copy()

# TO draw the contours, First argument is image we want to draw on, second is list of contours
# Next we have the contour index -> -1 means we want to draw all (we will supply index i for i'th contour)
# Example code for drawing the first, second and third contour.

# cv2.drawContours(coins, cnts, 0, (0, 255, 0), 2)
# cv2.drawContours(coins, cnts, 1, (0, 255, 0), 2)
# cv2.drawContours(coins, cnts, 2, (0, 255, 0), 2)

# Fourth argument: color of line we are going to draw - we pick green here.
# Last argument - thickness of the line we are going with. We draw it with 2 pixel thickness

cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
cv2.imshow("Coins", coins)
cv2.waitKey(0)

# CROPPING EACH COIN OUT

#cv2 bounding rectangle to find enclosing box to fit our contour. It allows us to crop it from the image
# It takes in a single parameter - contour and returns tuple of x,y position - starting, width and height
# We crop the coin out in a box. We then initialize a mask with the same width and height of our original image
# We call minEnclosingCircle that fits a circle to our contour, we pass in a circle variable, the current contour
# given out x,y coordinates of a circle along with the radius.
# Using the x,y and radius - draw circle on our mask representing the coin and crop the mask in the same manner.
# We crop out mask in same manner as the cropped coin.
# To ignore background, we make AND for coin image and mask coin.

for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    print("Coin #{}".format(i + 1))
    coin = image[y:y + h, x:x + w]
    cv2.imshow("Coin", coin)
    mask = np.zeros(image.shape[:2], dtype = "uint8")
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y + h, x:x + w]
    cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask =mask))
    cv2.waitKey(0)
