import cv2
import numpy as np
import sleepy_driver as ud
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
r_eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_righteye_2splits.xml')
l_eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_lefteye_2splits.xml')
open_closed = cv2.CascadeClassifier('cascade_files/haarcascade_openclosed_eyes.xml')


def get_face(frame):
    return face_cascade.detectMultiScale(frame,
                                  scaleFactor=1.1,
                                  minNeighbors=5,
                                  minSize=(30, 30))

def get_eyes2(frame):
    return open_closed.detectMultiScale(frame,
                                 scaleFactor=1.1,
                                 minNeighbors=5,
                                 minSize=(30, 30))

def get_eyes(gray):
    return l_eye_cascade.detectMultiScale(gray),r_eye_cascade.detectMultiScale(gray)


def eye_prediction(frame,model, eye, prediction_list, eye_side):
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
        ud.count += 1
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
                ud.right_eye_counter["right_eye_open"] += 1

            if prediction.all() == 0:
                right_eye_label = 'Closed'
                ud.right_eye_counter["right_eye_closed"] += 1
        if eye_side == "left":
            if prediction.all() == 1:
                left_eye_label = 'Open'
                ud.left_eye_counter["left_eye_open"] += 1

            if prediction.all() == 0:
                left_eye_label = 'Closed'
                ud.left_eye_counter["left_eye_closed"] += 1
    return prediction
