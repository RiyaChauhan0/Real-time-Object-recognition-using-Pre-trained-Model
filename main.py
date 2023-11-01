import numpy as np
import imutils
import time
import cv2

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
confThresh = 0.2        #threshold or a confidence level - that if there is any object then only it will going to detect it
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "ball", "bat", "pen"]
COLORS = np.random.uniform(0,255, size=(len(CLASSES),3))    #Randomly change the color for the classes based on it's length

print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)      #load the model file inside the DNN with Computer Vision
print("Model Loaded")
print("Starting Camera Feed...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _,frame = vs.read()
    frame = imutils.resize(frame, width=500)    #resizing the frame of the camera

    (h,w) = frame.shape[:2]     #give us the height & width of the frame useful for drwing the rectangle or writing the name of the object
    imgResizeBlob = cv2.resize(frame, (300,300))        #resizing lf input image - 300x300 (standard for MobileNetSSD)
    blob = cv2.dnn.blobFromImage(imgResizeBlob, 0.007843,(300,300),127.5)       #Converting the img into blob

    net.setInput(blob)  #passing blob as an input in the pre-trained model i.e loaded as net
    detections = net.forward()  #proceed img further for our classifications such as the class or accuracy or the bounding box
    detShape = detections.shape[2]   #we need shape to identify or have iterations between the loop
    for i in np.arange(0,detShape):
        confidence = detections[0,0,i,2]    #obtainig the confidence level - to identify if there's any object or not (i - for every img)
        if confidence > confThresh:
            idx = int(detections [0,0,i,1])     #idx is a class no. 
            box = detections [0,0,i,3:7]*np.array([w,h,w,h])    #to draw the bounding box of a class, 3:7 - means value from 3 to 6 i.e all the four values to draw the box
            (startX, startY, endX, endY) = box.astype("int")    #we got the four coordinates/values of box in the form of array so we have to convert it into integer

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)  #to get the class and confidence in %
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            if startY - 15 >15:
                y = startY-15
            else:
                y = startY+15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord("q"):
        break
vs.release()
cv2.destroyAllWindows()
    
