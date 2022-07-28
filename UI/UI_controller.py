import cv2

def draw_face(frame,faces):
    # cv2.rectangle(frame, (0, height - 50), (450, height), (0, 0, 0), thickness=cv2.FILLED)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def draw_eyes(frame,eyes):

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

def put_drowsiness_score(frame,text,height):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(frame, text, height, font, 1, (255, 255, 255), 1,cv2.LINE_AA)

def put_alert_text(frame):
    """
    Put text on image when an alarm is playing to wake up the driver
    :param frame: the frame to be processed and where the text will be put
    :return:
    """
    text = 'WAKE UP!'
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 0, 255)
    lineType = 2
    text_width, text_height = cv2.getTextSize(text, font, fontScale, lineType)[0]
    CenterCoordinates = (int(frame.shape[1] / 2) - int(text_width / 2), int(frame.shape[0] / 2) - int(text_height / 2))

    return cv2.putText(frame, text, CenterCoordinates, font, fontScale, fontColor, lineType)
