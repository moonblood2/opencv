import numpy as np
import urllib.request
import os
import cv2

def imurl(url):
    url_response = urllib.request.urlopen(url)
    img_array = np.array(bytearray(url_response.read()),dtype=np.uint8)
    return cv2.imdecode(img_array,-1)

url1 = 'https://pythonprogramming.net/static/images/opencv/3D-Matplotlib.png'
img1 = imurl(url1)
url2 = 'https://pythonprogramming.net/static/images/opencv/mainsvmimage.png'
img2 = imurl(url2)
url3 = 'https://pythonprogramming.net/static/images/opencv/mainlogo.png'
img3 = imurl(url3)

print("hello github")
print("I ain't tired")

cv2.imshow('img1',img1+img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
