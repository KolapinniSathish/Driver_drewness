import cv2 
import tensorflow as tf
from keras.models import load_model
import time 
import numpy as np

model=load_model("model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')

capture=cv2.VideoCapture(0)
while True:
    rate,frame=capture.read()
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    face = face_cascade.detectMultiScale(gray,minNeighbors = 3,scaleFactor = 1.1,minSize=(25,25))
    eyes = eye_cascade.detectMultiScale(gray,minNeighbors = 3,scaleFactor = 1.1,minSize=(25,25))
    
    

    for (fx,fy,fw,fh) in face:
        cv2.rectangle(gray,(fx,fy),(fx+fw,fy+fh),(0,255,0),2)
    for (x,y,w,h) in eyes:
        eye = frame[y:y+h,x:x+w]
        #eye = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(eye,(80,80))
        eye = eye/255
        eye = eye.reshape(80,80,3)
        eye = np.expand_dims(eye,axis=0)
        prediction = model.predict(eye)
        print(prediction)

    

    

    cv2.imshow("Video",gray)

    if cv2.waitKey(33)& 0xFF==ord('q'):
        break 
capture.release()
cv2.destroyAllWindows()