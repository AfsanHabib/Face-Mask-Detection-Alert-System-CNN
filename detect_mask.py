import os
import time
import warnings
from os.path import dirname, join

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

warnings.filterwarnings('ignore')
import json
import smtplib
import tkinter
import urllib.request
from email.message import EmailMessage
from tkinter import messagebox
from urllib.request import urlopen

# load our serialized face detector model from disk
prototxtPath = r"deploy.protext"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")



#_____Tkinter___________
root=tkinter.Tk()
root.withdraw()

#____Location__________
url='https://ipinfo.io/json'
response=urlopen(url)
data=str(json.load(response))




def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    # print(detections.shape)

    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # convert it from BGR to RGB channel and ordering, resize
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)





# initialize the video stream
print("CAMERA Starting......")
vs = VideoStream(src=0).start()


without_mask_countdown=0
# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600, height=600)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        if  mask<withoutMask:
            cv2.putText(frame, str("No mask: ")+str(without_mask_countdown), (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            
#             cv2.putText(frame,'without_mask_countdown: '+str(without_mask_countdown),(100,90),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(50,50,255),
#                        thickness=1,lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (50,50,255), 2)
            
            
            without_mask_countdown=without_mask_countdown+1
            
            if (without_mask_countdown>16):
                
                server =smtplib.SMTP('smtp.gmail.com', 587) #SMTP = Simple Mail Transfer Protocol & 587 is port
                server.starttls() # To secure TRusted server
                server.login('your mail ID', 'your mail Pass')
                email = EmailMessage()
                email['From'] = 'facemask.uctc@gmail.com'
                email['To'] ='afsan.uct@gmail.com,afsan.self@gmail.com'
                email['Subject'] =  'Face Mask Detection' 
                email.set_content('A person has been detected with out mask in the UCTC Computer Lab Area. Please Alert the Authoroties............!\n\n\n'+data)
                server.send_message(email)  
                
                messagebox.showwarning("Warning", "Message Sent..")
                
                without_mask_countdown=0
                
                



        else:
            cv2.putText(frame, str("Mask"), (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
            
            without_mask_countdown=0
            
            
    
    cv2.imshow("Result",frame)
    key=cv2.waitKey(1)
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
