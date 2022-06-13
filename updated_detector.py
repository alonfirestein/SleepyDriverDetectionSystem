from datetime import datetime
import cv2
import os
from keras.models import load_model
import numpy as np
import pandas as pd
import playsound
import os.path
from csv import writer
import traceback
import time


# Importing haar_cascade_files for face and eye classifier
# https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
# https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
r_eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_righteye_2splits.xml')
l_eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_lefteye_2splits.xml')
open_closed = cv2.CascadeClassifier('cascade_files/haarcascade_openclosed_eyes.xml')


labels = ['Closed', 'Open']
model = load_model('models/drowsiness_detector_model.h5')
path = os.getcwd()


# Global variables
camera_open = True
ALARM_ON = False
captured_photo = False
closed_eyes_threshold = 5
count, drowsiness_score, alarm_activated_counter = 0, 0, 0
right_eye_counter = {"right_eye_open": 0, "right_eye_closed": 0}
left_eye_counter = {"left_eye_open": 0, "left_eye_closed": 0}


def start_camera(URL):
    """
    Start the camera and return the capture object
    :param URL: the URL of the camera to be used (0 for laptop webcam)
    :return: the capture object
    """
    cap = cv2.VideoCapture(URL)
    # Check if the camera is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open camera")

    return cap


def update_database(file_path, started, ended, timer):
    """
    Update the database with the details of the latest drowsiness detection results
    :param file_path: the path of the file to be updated
    :param started: the time when the program started
    :param ended: the time when the program ended
    :param timer: a timer object
    :return:
    """
    global right_eye_counter, left_eye_counter, alarm_activated_counter
    # Total program duration in seconds
    total_duration = float("{0:.4f}".format(time.time() - timer))
    # Update database from last run
    columns = ["started",
               "ended",
               "total_duration",
               "right_eye_open",
               "right_eye_closed",
               "left_eye_open",
               "left_eye_closed",
               "alarm_activated_counter"]

    data = [started,
            ended,
            total_duration,
            right_eye_counter["right_eye_open"],
            right_eye_counter["right_eye_closed"],
            left_eye_counter["left_eye_open"],
            left_eye_counter["left_eye_closed"],
            alarm_activated_counter
            ]

    # Printing summary of the run to the console
    print("Drowsiness Summary:")
    for col, val in zip(columns, data):
        print(f"{col}: {val}")

    if not os.path.isfile(file_path):
        df = pd.DataFrame([data], columns=columns)
        df.to_csv(file_path)

    else:
        with open(file_path, 'a', newline='') as csv_file:
            writer_object = writer(csv_file)
            writer_object.writerow(data)


def eye_prediction(frame, eye, prediction_list, eye_side):
    """
    Predict the eye state of the face (open or closed) using my trained CNN model
    :param frame: the frame to be processed
    :param eye: the eye classifier used to predict the eye state (using haar cascade)
    :param prediction_list: the list to store the predictions
    :param eye_side: which eye is being predicted (left or right)
    :return:
    """
    global left_eye_counter, right_eye_counter, count
    prediction = prediction_list
    for (x, y, w, h) in eye:
        eye = frame[y:y + h, x:x + w]
        count += 1
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(eye, (24, 24))
        eye = eye / 255
        eye = eye.reshape(24, 24, -1)
        eye = np.expand_dims(eye, axis=0)
        predict_x = model.predict(eye)
        prediction = np.argmax(predict_x, axis=1)
        if eye_side == "right":
            if prediction.all() == 1:
                right_eye_label = 'Open'
                right_eye_counter["right_eye_open"] += 1

            if prediction.all() == 0:
                right_eye_label = 'Closed'
                right_eye_counter["right_eye_closed"] += 1
        if eye_side == "left":
            if prediction.all() == 1:
                left_eye_label = 'Open'
                left_eye_counter["left_eye_open"] += 1

            if prediction.all() == 0:
                left_eye_label = 'Closed'
                left_eye_counter["left_eye_closed"] += 1
    return prediction


def update_drowsiness_score(frame, right_eye_prediction, left_eye_prediction, height):
    """
    Update the drowsiness score based on the eye state (open or closed) and print it on the live video of the driver
    :param frame: the frame to be processed
    :param right_eye_prediction: the prediction of the right eye
    :param left_eye_prediction: the prediction of the left eye
    :param height: the height of the frame in order to print the drowsiness score on a good position
    :return:
    """
    global drowsiness_score
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(frame, 'Drowsiness Score:' + str(drowsiness_score), (175, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if right_eye_prediction[0] == 0 and left_eye_prediction[0] == 0:
        drowsiness_score += 1
        cv2.putText(frame, "Eyes: Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        drowsiness_score -= 1
        cv2.putText(frame, "Eyes: Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if drowsiness_score < 0:
        drowsiness_score = 0


def put_alert_text(img):
    """
    Put text on image when an alarm is playing to wake up the driver
    :param img: the image to be processed and where the text will be put
    :return:
    """
    text = 'WAKE UP!'
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 0, 255)
    lineType = 2
    text_width, text_height = cv2.getTextSize(text, font, fontScale, lineType)[0]
    CenterCoordinates = (int(img.shape[1] / 2) - int(text_width / 2), int(img.shape[0] / 2) - int(text_height / 2))

    return cv2.putText(img, text, CenterCoordinates, font, fontScale, fontColor, lineType)


def detection(cap):
    """
    The main function of the program. It is responsible for detecting the drowsiness of the driver.
    :param cap: the capture object used to capture the video of the driver
    :return:
    """
    global camera_open, ALARM_ON, drowsiness_score, right_eye_counter,\
           left_eye_counter, alarm_activated_counter, captured_photo
    started, timer = datetime.now().strftime('%Y-%m-%d %H:%M:%S'), time.time()

    while camera_open:
        right_eye_prediction, left_eye_prediction = [99], [99]
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        height, width = frame.shape[:2]
        # Converting frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Performing face detection, each face returns an array [x, y, width, height] of the bottom left corner
        # coordinates of the face, in addition to the width and height.
        faces = face_cascade.detectMultiScale(frame,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(30, 30))

        left_eye = l_eye_cascade.detectMultiScale(gray)
        right_eye = r_eye_cascade.detectMultiScale(gray)

        # Draw a small rectangle for the scores in the bottom left corner
        cv2.rectangle(frame, (0, height - 50), (450, height), (0, 0, 0), thickness=cv2.FILLED)

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

        # Predict the eye state of the face (open or closed)
        left_eye_prediction = eye_prediction(frame, left_eye, left_eye_prediction, eye_side="left")
        right_eye_prediction = eye_prediction(frame, right_eye, right_eye_prediction, eye_side="right")

        # Updating the drowsiness score and also printing it on the live video of the driver
        update_drowsiness_score(frame, right_eye_prediction, left_eye_prediction, height)

        # If we detect that the driver is sleepy, we play the alarm to wake them up
        if drowsiness_score >= closed_eyes_threshold:
            alarm_activated_counter += 1
            try:
                playsound.playsound("alarms/alarm_0.25.wav")
                put_alert_text(frame)
                # Capturing a photo of the sleepy driver as proof of their drowsiness
                if not captured_photo:
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cv2.imwrite(os.path.join(path, f"images/sleeping_driver-{current_time}.jpg"), frame)
                    captured_photo = True

            except Exception as e:
                traceback.print_exc()

        cv2.imshow('Drowsiness Detector', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    ended = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Updating database with all the information about the last session
    update_database("drowsiness_detection_data.csv", started, ended, timer)
    camera_open = False


