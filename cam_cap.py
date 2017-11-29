import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
##BufferedImage img = ImageIO.read(new ByteArrayInputStream(bytes));
blank = np.zeros((480,640*2,3),np.uint8)

time1 = time.time()
f = 0
while 1 :
    f+=1
    ret, frame = cap.read()
    blank[0:480,0:640] = frame
    blank[0:480,640:] = frame
    cv2.imshow('frame',blank)
    cv2.imwrite('1.jpg',blank)
##    print(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('fps:',f/(time.time()-time1))

cap.release()
cv2.destroyAllWindows()
