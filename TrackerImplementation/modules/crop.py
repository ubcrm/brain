import cv2
import numpy as np

def crop(frame, contour_corners):
    ''' Takes in a cv2 frame and iutputs the same image, masked to show only
        the region within the contour defined by countour_corners'''
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour_corners.astype(np.int32)], -1, 255, -1)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    return frame

