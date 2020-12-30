import cv2
import numpy as np
from modules import *

FIELD_DIMS = (8.08, 4.48)

# Different types of trackers, will be used to compare performances for tracking
TrDict = {'csrt': cv2.TrackerCSRT_create,
          'kcf': cv2.TrackerKCF_create,
          'boosting': cv2.TrackerBoosting_create,
          'mil': cv2.TrackerMIL_create,
          'tld': cv2.TrackerTLD_create,
          'medianflow': cv2.Tracker,
          'mosse': cv2.TrackerMOSSE_create}

trackers = cv2.MultiTracker_create() # Decide which tracker to
v = cv2.VideoCapture(r'C:\Users\sayem\Downloads\Clip2.mp4') # Find video to capture
fps = v.get(cv2.CAP_PROP_FPS) # Get the frame rate of video (later used to find velocity of robots)

time_elapsed = 1/fps

pts = np.zeros((4,2), np.int)
counter = 0

def mouseclicks(event,x,y,flags, parameters):
    global counter, pts, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        pts[counter] = x,y
        cv2.circle(frame,(x,y), 3, (0, 255, 0), -1)
        loc = x,y
        cv2.putText(frame, str(loc), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        counter = counter + 1
        print(pts)

ret, frame = v.read()
cv2.imshow('Frame', frame)
cv2.setMouseCallback("Frame", mouseclicks)
cv2.waitKey(0)

height, width = FIELD_DIMS
new_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
transform_matrix = cv2.getPerspectiveTransform(pts.astype(np.float32), new_corners)

k = 2
for i in range(k):
    cv2.imshow('Frame', frame)
    bbi = cv2.selectROI('Frame', frame)
    tracker_i = TrDict['csrt']()
    trackers.add(tracker_i, frame, bbi)

(success, boxes) = trackers.update(frame)

while True: #   Keep reading frames
    prev_boxes = boxes
    ret,frame = v.read() #read every frame in video
    if not ret:
        break #this is when the stream ends :(
    (success, boxes) = trackers.update(frame)
    
    for index, box in enumerate(boxes):
        (x,y,w,h) = [int(a) for a in box] #returns the bottom corner of the tracking box along with the width and height of the box
        (cpx, cpy) = int(x+w/2), int(y+h/2)
        cx, cy, cz = transform_matrix @ np.array([cpx, cpy, 1])
        # coords.append([cx / cz, cy / cz])
        # (vx,vy) = (x+w/2 - (prev_boxes[index,0]+prev_boxes[index,2]/2))/time_elapsed, (y+h/2 - (prev_boxes[index,1]+prev_boxes[index,3]/2))/time_elapsed
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #x,y is coordinates of the bottom left corner of the box while the other point is the top left corner of the box
        cv2.circle(frame,(cpx,cpy), 3, (0, 255, 0), -1)
        loc = round(cx/cz,2),round(cy/cz,2)
        cv2.putText(frame, str(loc), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

v.release()
cv2.destroyAllWindows()