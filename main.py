from camera import start_camera, detect_eyes


if __name__ == '__main__':
    # 0 is for laptop webcam for testing only, for deployment we will use phone camera IP
    # URL = "http://192.168.14.167:8080/video"
    URL = 0
    cap = start_camera(URL)
    detect_eyes(cap)
