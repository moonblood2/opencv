import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("C:\\Users\\ASUS\\Desktop\\opencv")
##img = cv2.imread('img.jpg')

##img = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)
##cv2.imshow('img',img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
##
##img = plt.imread('img.jpg')
##plt.imshow(img, cmap='gray',interpolation='quadric')
##plt.show()
##
##cv2.imwrite('0img.jpg',img)

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('cam.avi',fourcc, 20.0,(640,480))

hls = cv2.COLOR_BGR2HLS
gry = cv2.COLOR_BGR2GRAY
hsv=cv2.COLOR_BGR2HSV

font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = cap.read()
    h,w,channels = frame.shape
    
    mframe = cv2.cvtColor(frame,hsv)
    text = 'h:'+str(h)+' w:'+str(w)
    cv2.putText(mframe,text,(10,20),font, 0.5, (255,255,255),1,cv2.LINE_AA)
##    cv2.circle(mframe,(int(w/2),int(h/2)),10, (255,255,255))
    cv2.imshow('frame',mframe)
##    out.write(mframe)
##    cv2.imshow('gray',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
