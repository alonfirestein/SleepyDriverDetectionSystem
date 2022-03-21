from datetime import datetime
import cv2
import os
from eye_detection import eye_aspect_ratio
from keras.models import load_model
import numpy as np
import pandas as pd
import playsound
import csv
import os.path
import traceback
import time

# Importing haar_cascade_files for face and eye classifier
# https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
r_eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_righteye_2splits.xml')
l_eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_lefteye_2splits.xml')
open_closed = cv2.CascadeClassifier('cascade_files/haarcascade_openclosed_eyes.xml')

labels = ['Closed', 'Open']
model = load_model('models/cnncat2.h5')
path = os.getcwd()

# Global variables
open_camera = True
ALARM_ON = False
closed_eyes_threshold = 5
count, score = 0, 0
right_eye_counter = {"right_eye_open": 0, "right_eye_closed": 0}
left_eye_counter = {"left_eye_open": 0, "left_eye_closed": 0}


def start_camera(URL):
    cap = cv2.VideoCapture(URL)
    # Check if the camera is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open camera")

    return cap


def update_database(file_path, columns, data):
    if not os.path.isfile(file_path):
        df = pd.DataFrame([data], columns=columns)
    else:
        df = pd.read_csv(file_path)
        df = df.append([data], ignore_index=True)

    df.to_csv(file_path)


def detection(cap):
    global open_camera, ALARM_ON, score, count, right_eye_counter, left_eye_counter
    started = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    while True:
        thicc = 2
        right_eye_prediction, left_eye_prediction = [99], [99]
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(30, 30))

        left_eye = l_eye_cascade.detectMultiScale(gray)
        right_eye = r_eye_cascade.detectMultiScale(gray)

        # Draw a small rectangle for the scores in the bottom left corner
        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        # Draw a rectangle around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        eyes = open_closed.detectMultiScale(frame,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30))
        # Draw red rectangles around detected eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Attempting to draw red circles around detected eyes - Couldn't figure out the coordinates yet... To be fixed
        # for (ex, ey, ew, eh) in eyes:
        #     radius = eh // 2
        #     center_coordinates = (int(x+w//2), int(y+ey//2))
        #     cv2.circle(frame, center_coordinates, radius, (0, 0, 255), 2)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            count = count + 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            predict_x = model.predict(r_eye)
            right_eye_prediction = np.argmax(predict_x, axis=1)
            if right_eye_prediction.all() == 1:
                right_eye_label = 'Open'
                right_eye_counter["right_eye_open"] += 1
            if right_eye_prediction.all() == 0:
                right_eye_label = 'Closed'
                right_eye_counter["right_eye_closed"] += 1
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            count = count + 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            predict_x = model.predict(l_eye)
            left_eye_prediction = np.argmax(predict_x, axis=1)

            if left_eye_prediction.all() == 1:
                left_eye_label = 'Open'
                left_eye_counter["left_eye_open"] += 1
            if left_eye_prediction.all() == 0:
                left_eye_label = 'Closed'
                left_eye_counter["left_eye_closed"] += 1
            break
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        if right_eye_prediction[0] == 0 and left_eye_prediction[0] == 0:
            score += 1
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score = score - 1
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score < 0:
            score = 0
        cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # If we detect that the driver is sleepy, we play the alarm to wake them up
        if score >= closed_eyes_threshold:
            # cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            try:
                playsound.playsound("alarms/alarm_0.25.wav")
            except Exception as e:
                traceback.print_exc()

            if thicc < 16:
                thicc += 2
            else:
                thicc -= 2
                if thicc < 2:
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        cv2.imshow('Drowsiness Detector', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            open_camera = False
            break

    cap.release()
    cv2.destroyAllWindows()
    ended = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Update database from last run
    columns = ["started",
               "ended",
               "right_eye_open",
               "right_eye_closed",
               "left_eye_open",
               "left_eye_closed"]

    data = [started,
            ended,
            right_eye_counter["right_eye_open"],
            right_eye_counter["right_eye_closed"],
            left_eye_counter["left_eye_open"],
            left_eye_counter["left_eye_closed"]
            ]

    print("Drowsiness Summary:")
    for col, val in zip(columns, data):
        print(f"{col}: {val}")
    update_database("drowsiness_detection_data.csv", columns, data)
