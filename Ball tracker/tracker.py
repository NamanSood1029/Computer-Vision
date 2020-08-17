# import arguments
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# We import deque (a list-like data structure) which can append and pop very quickly
# This will be used to store the past N (x,y) coordinates of the ball -> draw contrails
# imutils will be used to make tasks like resizing much eaiser

# arguments for our function
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='Path to the video file (optional)')
ap.add_argument('-b', '--buffer', type=int, default=64, help='max buffer size')
args = vars(ap.parse_args())

# upper and lower boundaries of the color green in the HSV color space
# determined using range-detector in imutils library
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args['buffer'])

# if no video path was supplied -> use web-cam
if not args.get('video', False):
    vs = VideoStream(src=0).start()

# get the reference otherwise
else:
    vs = cv2.VideoCapture(args['video'])

# allow the camera/file to warm up
time.sleep(2.0)

while True:
    # it returns a 2-tuple. The first entry (grabbed) indicates whether it was read or not
    # the second argument is the video frame itself

    frame = vs.read()

    # handle method if video frame from a video or a live-stream
    frame = frame[1] if args.get('video', False) else frame

    # if we are reading from a video file and the frame is not read
    # it infers that the video has ended and we can break from the while loop
    if frame is None:
        break

    # resize frame, blur and convert to HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the green color
    # perform series of dilation and erosion to remove any small blobs
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize (x,y) to the center
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # proceed only if a contour was found
    if len(cnts) > 0:
        # find largest contour (perimeter) and use it to compute the minimum enclosing circle and the center
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if radius > 5:
            # draw circle and centroid on the frame and update tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    pts.appendleft(center)
    # loop over the tracked points
    for i in range(1, len(pts)):
        # if none, ignore them
        if pts[i-1] is None or pts[i] is None:
            continue

        # otherwise, compute thickness of the line and draw connecting lines
        thickness = int(np.sqrt(args['buffer']/ float(i+1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# if we are not using a video file, stop the video stream
if not args.get('video', False):
    vs.stop()

# otherwise, release the camera  
else:
    vs.release()

cv2.destroyAllWindows()