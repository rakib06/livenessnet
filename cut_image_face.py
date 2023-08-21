from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from imutils import paths

args ={
    "dataset": "",
    "detector": "face_detector",
    }

protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)



imagePaths =  list(paths.list_images(args["dataset"]))

for imagePath in imagePaths:
    # extract the class label from the filename, load the image and
    # resize it to be a fixed 32x32 pixels, ignoring aspect ratio
    label = imagePath.split(os.path.sep)[-2]
    # print(": ", imagePath.split(os.path.sep)[1])
    # print("os.path.sep: ", os.path.sep)
    # print("label: ", label)
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (256, 256))
    
    
    frame = imutils.resize(image, width=600)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
  
    # prediction
    confidence = detections[0, 0, 0, 2]
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for
        # the face and extract the face ROI
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # ensure the detected bounding box does fall outside the
        # dimensions of the frame
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)

        # extract the face ROI and then preproces it in the exact
        # same manner as our training data
        face = frame[startY:endY, startX:endX]
        try:
            FACE_IMAGE_PART = cv2.resize(face, (256, 256))
            os.makedirs('face') if not os.path.exists('face') else ''
            timestamp = int(time.time())  # Generate a unique timestamp
            image_filename = f"fake/face_{timestamp}.jpg"  # Create a filename with timestamp
            cv2.imwrite(image_filename, FACE_IMAGE_PART)
        except:
            pass

    # update the data and labels lists, respectively
    # data.append(image)
    # labels.append(label)
