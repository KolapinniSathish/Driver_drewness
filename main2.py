import cv2
import time
import numpy as np
from playsound import playsound
from keras.models import load_model
e=1
alaram="horror_sms.mp3"
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')  # Load the eye cascade classifier
face_casecade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
model=load_model("model.h5")
time_count=[]
start_time=0
while e>=1:
    vech_start=input("\nPlease Enter 0 for Exit! \nIs vachile Engine on [YES/NO]?:")
    if vech_start.lower()=="yes" or vech_start.lower()=='y':
        vech_motion=input("Is vachile in motion [YES/NO]?:")
        if vech_motion.lower()=="yes" or vech_motion.lower()=='y':
            cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the camera index if you have multiple cameras.
            while True:
                ret, frame = cap.read()  # Read a frame from the camera
                frame=cv2.flip(frame,1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if not ret:
                    break
                faces = face_casecade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                



                for (x,y,w,h) in faces:
                    face = frame[y:y+h,x:x+w]
                    cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),3)
                    face = cv2.resize(face,(80,80))
                    face=face/255
                    face = face.reshape(80,80,3)
                    face = np.expand_dims(face,axis=0)
                    prediction = model.predict(face)
                    print(prediction)


                    

                cv2.imshow('Eye Detection2', gray)
                if cv2.waitKey(33) & 0xFF == 27:  # Press 'Esc' to exit
                    break

                if len(faces)==0:
                    start_time=time.time()
                    time_count.append(start_time)  
                else:
                    time_count.clear()
                    start_time=0
                if len(time_count)==0:
                    continue
                elif time_count[-1]-time_count[0]>=5:
                    print("No eyes detected!")
                    playsound(alaram)
                else:
                    print(time_count[-1]-time_count[0])
                
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("Detecter will start when vachile in mothion!")
    
    elif vech_start=='0':
        e=0 
    else:
        print("As per your input vachile engine detected as OFF!")