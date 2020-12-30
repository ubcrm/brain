import cv2
import numpy as np

def mouseclicks(event,x,y,flags, parameters):
    global counter, pts, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        pts[counter] = x,y
        cv2.circle(frame,(x,y), 3, (0, 255, 0), -1)
        loc = x,y
        cv2.putText(frame, str(loc), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        counter = counter + 1
        print(pts)