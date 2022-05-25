from camera import start_camera, detect_eyes
from updated_detector import start_camera, detection
from eye_detection import eye_aspect_ratio

if __name__ == '__main__':
    # 0 is for laptop webcam for testing only, for deployment we will use phone camera IP
    # URL = "http://192.168.14.167:8080/video"
    URL = 0
    cap = start_camera(URL)
    detection(cap)