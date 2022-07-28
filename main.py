from sleepy_driver import start_camera, run


if __name__ == '__main__':
    # 0 is for laptop webcam for testing only, for deployment we will use phone camera IP
    # URL = "http://192.168.14.167:8080/video"
    URL = 0
    cap = start_camera(URL)
    run(cap)