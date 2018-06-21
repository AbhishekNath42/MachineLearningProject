import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cam = cv2.VideoCapture(0)

while True:
    ret,img = cam.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h),(255,0,0),2)
        roig= gray_img[y:y+h, x:x+w]
        roic= img[y:y+h, x:x+w]
        eyes=eye_cascade.detectMultiScale(roig)
        for(a,b,c,d) in eyes:
            cv2.rectangle(roic, (a,b), (a+c,b+d),(0,255,0),2)

    cv2.imshow('fed',img)
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
