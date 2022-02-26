import os
import numpy as np
import cv2
import playsound


# Importing haar cascade files for face and eye classifier
# https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
open_closed = cv2.CascadeClassifier('cascade_files/haarcascade_openclosed_eyes.xml')

# Global variables
open_camera = True
closed_eyes_threshold = 5
ALARM_ON = False


def start_camera(URL):
    cap = cv2.VideoCapture(URL)
    # Check if the camera is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open camera")

    return cap


def detect_eyes(cap):
    global open_camera, ALARM_ON
    closed_eyes_counter, opened_eyes_counter = 0, 0
    while open_camera:
        ret, img = cap.read()
        if ret:
            # Convert frame to grayscale to identify face better
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image
            face = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_identified = len(face) > 0
            if face_identified:
                # Draw a rectangle around the face to focus on it
                for (x, y, w, h) in face:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                frame_tmp = img[face[0][1]:face[0][1] + face[0][3], face[0][0]:face[0][0] + face[0][2]:1, :]
                frame = frame[face[0][1]:face[0][1] + face[0][3], face[0][0]:face[0][0] + face[0][2]:1]
                eyes = open_closed.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(eyes) == 0:
                    closed_eyes_counter += 1
                    print('ALERT: Closed eyes detected')  # Print method is only for testing

                else:
                    opened_eyes_counter += 1
                    print('Open eyes detected :)')  # Print method is only for testing

                print(f"Closed: {closed_eyes_counter}, Open: {opened_eyes_counter}")
                # If our closed eyes counter meets the threshold: We play the alarm to wake up the driver!
                if closed_eyes_counter == closed_eyes_threshold:
                    ALARM_ON = True

                # If the driver is awake for consecutive countings, we reinitialize the counters
                # and turn off the alarm if it's on
                if opened_eyes_counter == 10:
                    closed_eyes_counter = 0
                    opened_eyes_counter = 0
                    ALARM_ON = False

                # Boolean flag to play alarm, sound will play until driver is awake for several consecutive seconds
                # meaning until the opened_eyes_counter meets its threshold to turn off the alarm
                if ALARM_ON:
                    # os.system("say beep")  # On mac: Says beep when eyes closed (funny, but delete for deployment)
                    playsound.playsound("alarms/alarm_0.5seconds.wav")
                    put_alert_text(img)

                frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Drowsiness Detector', frame_tmp)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            open_camera = False
            break

    cap.release()
    cv2.destroyAllWindows()


# Put text on image when an alarm is playing to wake up the driver
def put_alert_text(img):
    # Text details
    text = 'WAKE UP!'
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 0, 255)
    lineType = 2
    text_width, text_height = cv2.getTextSize(text, font, fontScale, lineType)[0]
    CenterCoordinates = (int(img.shape[1] / 2) - int(text_width / 2), int(img.shape[0] / 2) - int(text_height / 2))

    return cv2.putText(img, text, CenterCoordinates, font, fontScale, fontColor, lineType)

