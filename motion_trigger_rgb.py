import os
import cv2
import time
import subprocess
from datetime import datetime
import configparser

LOCK_FILE = "/tmp/thermal_capture.lock"

def is_thermal_running():
    return os.path.exists(LOCK_FILE)

def create_lock():
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

def remove_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

def write_log(info, message):
    now = time.strftime('%Y/%m/%d %H:%M:%S')
    log_msg = f'{info} {now} : {message}'
    print(log_msg)
    with open('thermal_outputs.log', 'a') as f:
        f.write(log_msg + '\n')

def load_config():
    config = configparser.ConfigParser()
    config.read('thermal.par')
    cam_name = config.get('Camera', 'cam_name')
    thresh = config.getint('Temperature', 'temp_trigger')
    duration = config.getint('Time', 'duration')
    output = config.get('Output', 'folder')
    return cam_name, thresh, duration, output

def find_Arducam():
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame.shape[1] == 3840 and frame.shape[0] == 2160:
                cap.release()
                return i
            cap.release()
    return 0

cam_name, motion_thresh, duration, output = load_config()
output_folder = os.path.join(output, f"{cam_name}_{datetime.now().strftime('%Y_%m_%d')}")
rgb_folder = os.path.join(output_folder, "RGB")
os.makedirs(rgb_folder, exist_ok=True)

cap = cv2.VideoCapture(find_Arducam(), cv2.CAP_V4L2)
ret, background = cap.read()
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background_float = background_gray.astype("float")

alpha = 0.05
cooldown_seconds = 10
last_trigger_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.accumulateWeighted(gray, background_float, alpha)
    background_gray = cv2.convertScaleAbs(background_float)

    center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
    box_size = 100
    roi = cv2.absdiff(background_gray, gray)
    roi_thresh = cv2.threshold(roi, 25, 255, cv2.THRESH_BINARY)[1]
    center_roi = roi_thresh[center_y - box_size:center_y + box_size,
                            center_x - box_size:center_x + box_size]
    motion_score = cv2.countNonZero(center_roi)

    if motion_score > motion_thresh and (time.time() - last_trigger_time > cooldown_seconds):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rgb_filename = os.path.join(rgb_folder, f"rgb_{timestamp}.jpg")
        cv2.imwrite(rgb_filename, frame)
        write_log("SAVE", f"RGB image saved: {rgb_filename}")

        if not is_thermal_running():
            write_log("INFO", "Motion detected — launching thermal capture script")
            create_lock()
            subprocess.Popen(['python3', 'thermal_capture.py'])
        else:
            write_log("INFO", "Motion detected — but thermal capture already running")

        last_trigger_time = time.time()

    time.sleep(0.1)
