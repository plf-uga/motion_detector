import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from libuvc_wrapper import *
import configparser
from queue import Queue
from ctypes import *

BUF_SIZE = 2
q = Queue(BUF_SIZE)

os.chdir("/mnt/external/get_thermal_data")

#----------------------- Util Functions -----------------------#
def find_Arducam():
    for i in range(10):
        if os.path.exists(f'/dev/video{i}'):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame.shape[:2] == (2160, 3840):
                    cap.release()
                    print(f"Saving Arducam as index {i}!")
                    return i
            cap.release()
    return 0

def write_log(info, message, verbose):
    now = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
    line = f'{info} {now} : {message}\n'
    with open('thermal_outputs.log', 'a') as f:
        if verbose:
            print(line.strip())
        f.write(line)

def ktoc(val):
    return (val - 27315) / 100.0

def raw_to_colored(data, min_temp, max_temp):
    data_c = ktoc(data)
    norm = np.clip((data_c - min_temp) / (max_temp - min_temp), 0, 1)
    norm = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_HOT)

def py_frame_callback(frame, userptr):
    ptr = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    arr = np.frombuffer(ptr.contents, dtype=np.uint16).reshape((frame.contents.height, frame.contents.width))
    q.put(arr)

def load_config():
    config = configparser.ConfigParser()
    config.read('thermal.par')
    return (
        config.get('Camera', 'cam_name'),
        config.getint('Camera', 'img_height'),
        config.getint('Camera', 'img_width'),
        config.getint('Temperature', 'temp_max'),
        config.getint('Temperature', 'temp_min'),
        config.getint('Temperature', 'temp_trigger'),
        config.getint('Time', 'duration'),
        config.getint('Camera', 'index'),
        config.get('Output', 'folder')
    )

#----------------------- Thermal Thread -----------------------#
class ThermalCaptureThread(threading.Thread):
    def __init__(self, duration, temp_min, temp_max, w, h, output_folder, motion_flag):
        super().__init__()
        self.running = threading.Event()
        self.running.set()
        self.duration = duration
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.w = w
        self.h = h
        self.output_folder = output_folder
        self.motion_flag = motion_flag
        self.callback = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

    def stop(self):
        self.running.clear()

    def run(self):
        ctx, dev, devh, ctrl = POINTER(uvc_context)(), POINTER(uvc_device)(), POINTER(uvc_device_handle)(), uvc_stream_ctrl()
        if libuvc.uvc_init(byref(ctx), 0) < 0:
            write_log("ERROR", "uvc_init error", 1)
            return

        try:
            if libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0) < 0:
                write_log("ERROR", "uvc_find_device error", 1)
                return
            if libuvc.uvc_open(dev, byref(devh)) < 0:
                write_log("ERROR", "uvc_open error", 1)
                return

            formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
                                                   formats[0].wWidth, formats[0].wHeight,
                                                   int(1e7 / formats[0].dwDefaultFrameInterval))

            libuvc.uvc_start_streaming(devh, byref(ctrl), self.callback, None, 0)
            write_log("INFO", "Thermal camera started", 1)
            start_time = time.time()

            while self.running.is_set() and time.time() - start_time < self.duration:
                try:
                    data = q.get(timeout=1)
                    if self.motion_flag['active']:
                        img = cv2.resize(data, (self.w, self.h))
                        img_colored = raw_to_colored(img, self.temp_min, self.temp_max)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        path = os.path.join(self.output_folder, "color_thermal", f"thermal_{ts}.jpg")
                        cv2.imwrite(path, img_colored)
                        write_log("SAVE", f"Saved thermal image: {path}", 1)
                except:
                    continue

            libuvc.uvc_stop_streaming(devh)
            write_log("INFO", "Thermal camera stopped", 1)
        finally:
            libuvc.uvc_exit(ctx)
            self.running.clear()

#------------------------- Main ------------------------------#
cam_name, h, w, temp_max, temp_min, thresh, duration, index, output = load_config()
index = find_Arducam()
cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

output_folder = os.path.join(output, f"{cam_name}_{datetime.now().strftime('%Y_%m_%d')}")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "color_thermal"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "RGB"), exist_ok=True)

write_log("INFO", f"Created directory: {output_folder}", 1)

ret, bg = cap.read()
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype("float")

alpha = 0.05
trigger_threshold = 5000
motion_flag = {'active': False}

thermal_thread = ThermalCaptureThread(duration, temp_min, temp_max, w, h, output_folder, motion_flag)
thermal_thread.start()

start_time = time.time()
while time.time() - start_time < duration:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.accumulateWeighted(gray, bg_gray, alpha)
    diff = cv2.absdiff(cv2.convertScaleAbs(bg_gray), gray)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    cx, cy, bs = gray.shape[1]//2, gray.shape[0]//2, 100
    roi = mask[cy - bs:cy + bs, cx - bs:cx + bs]
    score = cv2.countNonZero(roi)

    if score > trigger_threshold:
        motion_flag['active'] = True
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(output_folder, "RGB", f"rgb_{ts}.jpg")
        cv2.imwrite(path, frame)
        write_log("SAVE", f"Saved RGB image: {path}", 1)
    else:
        motion_flag['active'] = False

    time.sleep(0.1)

thermal_thread.stop()
thermal_thread.join()
cap.release()
cv2.destroyAllWindows()
write_log("INFO", "Main program ended", 1)
