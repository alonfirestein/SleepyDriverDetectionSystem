
from datetime import datetime
import cv2
import os
import pandas as pd
path = os.getcwd()
def save_drowsiness_img(frame):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.imwrite(os.path.join(path, f"images/sleeping_driver-{current_time}.jpg"), frame)


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
