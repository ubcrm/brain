import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)

# Add hard coded cropping boundaries to reduce the number of corners detected outside the arena
# If any of the parameters are Nonetype then the functionality reverts to default
bounds = None
ret = args.image.split('\\')[-1]
if '1' in ret:
    bounds = np.array([[187,250],[700,205],[1746,626],[817,1076],[745,1071]])
elif '2' in ret:
    bounds = np.array([[123, 144],[645, 88],[1704, 519],[779, 986]])

x = None
y = None
w = None
h = None
if bounds is not None:
    rect = cv.boundingRect(bounds)
    x,y,w,h = rect
    bounds = bounds - bounds.min(axis=0)

    
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.2,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()

# Crop out the first frame for corner detection
if x is not None and bounds is not None:
        cropped = old_frame[y:y+h, x:x+w].copy()
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv.drawContours(mask, [bounds], -1, (255, 255, 255), -1, cv.LINE_AA)
        old_frame = cv.bitwise_and(cropped, cropped, mask=mask)

# cv.imshow('img', old_frame)
# cv.waitKey(0)  
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()

    # Crop out all frames with arena boundary
    if x is not None and bounds is not None:
        cropped = frame[y:y+h, x:x+w].copy()
        cropmask = np.zeros(cropped.shape[:2], np.uint8)
        cv.drawContours(cropmask, [bounds], -1, (255, 255, 255), -1, cv.LINE_AA)
        frame = cv.bitwise_and(cropped, cropped, mask=cropmask)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)

    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)