import cv2
import numpy as np

from modules import *

frame = crop(np.zeros((200,200)),np.array([[0,0],[100,0],[0,100],[100,100]]))

cv2.imshow('frame',frame)