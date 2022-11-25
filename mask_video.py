import numpy as np
import os
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
cap=cv2.VideoCapture(0)
faceproto='opencv_face_detector.pbtxt'
facemodel='opencv_face_detector_uint8.pb'
faceNet=cv2.dnn.readNet(facemodel,faceproto)
model=load_model('mobilenet_classifier.h5')
def detect(faceNet,frame):
    framewidth=frame.shape[0]
    frameheight=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame,1.0,(224,224),[104,117,123],swapRB=False)
    faceNet.setInput(blob)
    detections=faceNet.forward()
    boxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>0.7:
            x1=int(detections[0,0,i,3]*frameheight)
            y1=int(detections[0,0,i,4]*framewidth)
            x2=int(detections[0,0,i,5]*frameheight)
            y2=int(detections[0,0,i,6]*framewidth)
            boxes.append([x1,y1,x2,y2])
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    return boxes
def predict(frame,model,boxes):
    for box in boxes:
        x,y,w,h=box
        roi_frame=frame[y:y+h,x:x+w]
        face=cv2.cvtColor(roi_frame,cv2.COLOR_BGR2RGB)
        face=cv2.resize(frame,(224,224))
        face=img_to_array(face)
        face=preprocess_input(face)
        face=np.expand_dims(face,axis=0)
        (mask,withoutMask)=model.predict(face)[0]
        
        #determine the class label and color we will use to draw the bounding box and text
        label='Mask' if mask>withoutMask else 'No Mask'
        color=(0,255,0) if label=='Mask' else (0,0,255)
        
        #include the probability in the label
        label="{}: {:.2f}%".format(label,max(mask,withoutMask)*100)
        
        #display the label and bounding boxes
        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(frame,(x,y),(w,h),color,2)
    return frame
while True:
    ret,frame=cap.read()
    boxes=detect(faceNet,frame)
    canvas=predict(frame,model,boxes)
    cv2.imshow('canavas',canvas)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
