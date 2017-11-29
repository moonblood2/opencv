import cv2
import numpy as np

img = cv2.imread('1.jpg',cv2.IMREAD_COLOR)

blank = np.zeros((480,640*2,3),np.uint8)

blank[0:480,0:640] = img
blank[0:480,640:] = img

cv2.imshow('img',blank)
