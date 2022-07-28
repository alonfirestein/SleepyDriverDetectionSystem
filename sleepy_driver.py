from datetime import datetime
import cv2
from detection import Predictors
from UI import UI_controller as ui
from keras.models import load_model
import playsound
import traceback
import time
from output_data import data_save


labels = ['Closed', 'Open']
model = load_model('models/drowsiness_detector_model.h5')


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
    text = 'Drowsiness Score:' + str(drowsiness_score)
    text_height = (175, height - 20)
    ui.put_drowsiness_score(frame,text,text_height)

    if right_eye_prediction[0] == 0 and left_eye_prediction[0] == 0:
        drowsiness_score += 1
        text = "Eyes: Closed"
        text_height = (10, height - 20)

    else:
        drowsiness_score -= 1
        text = "Eyes: Open"
        text_height = (10, height - 20)

    ui.put_drowsiness_score(frame,text,text_height)

    if drowsiness_score < 0:
        drowsiness_score = 0


def run(cap):
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
        frame = cv2.resize(frame, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]


        # Converting frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Performing face detection, each face returns an array [x, y, width, height] of the bottom left corner
        # coordinates of the face, in addition to the width and height.

        faces = Predictors.get_face(frame)
        left_eye,right_eye = Predictors.get_eyes(gray)

        # Draw a small rectangle for the scores in the bottom left corner
        ui.draw_face(frame,faces)
        eyes = Predictors.get_eyes2(frame)

        # Draw red rectangles around detected eyes
        ui.draw_eyes(frame,eyes)


        # Predict the eye state of the face (open or closed)
        left_eye_prediction = Predictors.eye_prediction(frame, model, left_eye, left_eye_prediction, eye_side="left")
        right_eye_prediction = Predictors.eye_prediction(frame, model, right_eye, right_eye_prediction, eye_side="right")

        # Updating the drowsiness score and also printing it on the live video of the driver
        update_drowsiness_score(frame, right_eye_prediction, left_eye_prediction, height)

        # If we detect that the driver is sleepy, we play the alarm to wake them up
        if drowsiness_score >= closed_eyes_threshold:
            alarm_activated_counter += 1
            try:
                playsound.playsound("alarms/alarm_0.25.wav")
                ui.put_alert_text(frame)
                # Capturing a photo of the sleepy driver as proof of their drowsiness
                if not captured_photo:
                    data_save.save_drowsiness_img(frame)
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
    data_save.update_database("drowsiness_detection_data.csv", started, ended, timer)
    camera_open = False
