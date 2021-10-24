from check import *
import cv2
from tensorflow import keras
import numpy as np
import tensorflow as tf
import PIL.Image



model = keras.models.load_model("facemask_model.h5")

database = {}

database["with mask"] = img_to_encoding("images/mask_new2.jpg", model)
database["without mask"] = img_to_encoding("images/nomask.jpg", model)
database["no face"] = img_to_encoding("images/home 2.jpg", model)

cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)


while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.flip(frame,1)
    frame1 = cv2.resize(frame, (299, 299))

    min_dist, identity = check_mask_live(frame1, database, model)
    string = "The person on the frame is " + str(identity).upper()

    if identity == "with mask":
        cv2.putText(frame, string, (300,40), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)

    elif identity == "without mask":
        cv2.putText(frame, string, (300,40), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

    else:
        string = "There is NO PERSON in the frame"
        cv2.putText(frame, string, (300,40),cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()