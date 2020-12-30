import numpy as np
import cv2

#imput: pts: points of the image I plan to transform. Follows the order: top-left -> top-right -> bottom-right -> bottom-left \
#       frame: frame on which we want to do the perspective transform
def four_point_transform(frame, pts):
    rect = np.zeros((4, 2), dtype="float32")
    rect = pts
