#New program to start the cameras based on the RGB camera
#Also fixes the device index problem

import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from libuvc_wrapper import *
import configparser
from queue import Queue

BUF_SIZE = 2
q = Queue(BUF_SIZE)

os.chdir("/mnt/external/get_thermal_data")

##########################################################################
#---------------Util Functions for the program---------------------------#
##########################################################################

def find_Arducam():
    device_info = []
    for i in range(10):
        device_path = f'/dev/video{i}'
        if os.path.exists(device_path):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    shape = frame.shape  # (height, width, channels)
                    h, w, c = shape
                    if w == 3840 and h == 2160 and c ==3:
                        index = i
                        print(f"Saving Arducam as index {i}!")
                else:
                   print(f"[WARN] Camera index {i} exists but could not be open")
                cap.release()
    return int(index)


def write_log(info, message, verbose):
    current_timestamp = time.time()
    # Convert the timestamp to the desired format
    current_time = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(current_timestamp))
    cout = f'{info} {current_time} : {message}  \n'
    with open('thermal_outputs.log', 'a') as file:
        if verbose == 1:
            print(cout)
        file.write(cout)


def display_temperature(img, val_k, loc, color):
    """
    Display the temperature at the given location on the image.
    """
    val = ktoc(val_k)
    cv2.putText(img, "{0:.1f} degC".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)




# Configuration
def load_config():
    config = configparser.ConfigParser()
    config.read('thermal.par')
    cam_name = config.get('Camera', 'cam_name')
    h = config.getint('Camera', 'img_height')
    w = config.getint('Camera', 'img_width')
    index = config.getint('Camera', 'index')
    temp_max = config.getint('Temperature', 'temp_max')
    temp_min = config.getint('Temperature','temp_min')
    thresh = config.getint('Temperature','temp_trigger')
    duration = config.getint('Time','duration')
    output = config.get('Output', 'folder')
    return cam_name, h, w, temp_max, temp_min, thresh, duration, index, output


def raw_to_colored(data, min_temp, max_temp):
    """
    Convert the raw thermal data to a standardized 8-bit colored image.

    Args:
        data: 16-bit raw thermal data.
        min_temp: The minimum temperature (in °C) for normalization.
        max_temp: The maximum temperature (in °C) for normalization.

    Returns:
        A colored image representing the normalized thermal data.
    """
    data_celsius = ktoc(data)
    data_normalized = np.clip((data_celsius - min_temp) / (max_temp - min_temp), 0, 1)
    data_normalized = (data_normalized * 255).astype(np.uint8)
    colored_img = cv2.applyColorMap(data_normalized, cv2.COLORMAP_HOT)
    return colored_img


# Convert raw data to temperature
def ktoc(val):
    return (val - 27315) / 100.0


# Callback function to receive frames from libuvc
def py_frame_callback(frame, userptr):
    frame_data = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(frame_data.contents, dtype=np.uint16).reshape((frame.contents.height, frame.contents.width))
    q.put(data)



# Thermal thread class
class ThermalCaptureThread(threading.Thread):
    def __init__(self, duration, temp_min, temp_max, w, h, output_folder):
        super().__init__()
        self.running = threading.Event()
        self.running.set()
        self.duration = duration
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.w = w
        self.h = h
        self.output_folder = output_folder
        self.PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

    def stop(self):
        self.running.clear()

    def run(self):
        ctx, dev, devh, ctrl = POINTER(uvc_context)(), POINTER(uvc_device)(), POINTER(uvc_device_handle)(), uvc_stream_ctrl()
        res = libuvc.uvc_init(byref(ctx), 0)
        if res < 0:
            write_log("ERROR", "uvc_init error", 1)
            return

        try:
            res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
            if res < 0:
                write_log("ERROR", "uvc_find_device error", 1)
                return

            res = libuvc.uvc_open(dev, byref(devh))
            if res < 0:
                write_log("ERROR", "uvc_open error", 1)
                return

            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            libuvc.uvc_get_stream_ctrl_format_size(
                devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
                frame_formats[0].wWidth, frame_formats[0].wHeight,
                int(1e7 / frame_formats[0].dwDefaultFrameInterval)
            )

            libuvc.uvc_start_streaming(devh, byref(ctrl), self.PTR_PY_FRAME_CALLBACK, None, 0)
            write_log("INFO", "Thermal camera started", 1)

            start_time = time.time()
            while self.running.is_set() and (time.time() - start_time < self.duration):
                try:
                    data = q.get(timeout=1)
                    data_resized = cv2.resize(data[:, :], (self.w, self.h))  # Use config-defined w and h
                    colored_img = raw_to_colored(data_resized, self.temp_min, self.temp_max)

                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(self.output_folder, "color_thermal", f"thermal_{timestamp}.jpg")
                    cv2.imwrite(filename, colored_img)
                    write_log("SAVE", f"Saved image: {filename}", 1)
                    time.sleep(1)
                except:
                    continue

            libuvc.uvc_stop_streaming(devh)
            write_log("INFO", "Thermal camera stopped", 1)

        finally:
            libuvc.uvc_exit(ctx)



################################################################################
################################################################################
index = find_Arducam()

cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

#Import program parameters
cam_name, h, w, temp_max, temp_min, thresh, duration, index, output = load_config()

# Ensure output folder exists
output_folder = os.path.join(output, f"{cam_name}_{datetime.now().strftime('%Y_%m_%d')}")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "color_thermal"), exist_ok = True)
os.makedirs(os.path.join(output_folder, "raw_thermal"), exist_ok = True)
rgb_folder = os.path.join(output_folder, "RGB")
os.makedirs(rgb_folder, exist_ok=True)  # <-- FIX
write_log(info = "INFO", message = f"Created directory: {output_folder}", verbose = 1)



#Initialize Background as float
ret, background = cap.read()
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background_float = background_gray.astype("float")

alpha = 0.05
trigger_threshold = 5000

motion_detected = False
last_motion_time = time.time()
thermal_thread = None
cooldown_seconds = 10

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.accumulateWeighted(gray, background_float, alpha)
    background_gray = cv2.convertScaleAbs(background_float)

    box_size = 100
    center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
    top_left_x = center_x - box_size // 2
    top_left_y = center_y - box_size // 2
    bottom_right_x = center_x + box_size // 2
    bottom_right_y = center_y + box_size // 2

    # Compute absolute difference
    diff = cv2.absdiff(background_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Crop the central ROI box where cow is expected
    roi = thresh[center_y-box_size:center_y+box_size, center_x-box_size:center_x+box_size]
    motion_score = cv2.countNonZero(roi)
    #print(motion_score)

    # Save RGB image if motion is above threshold
    if motion_score > trigger_threshold:
        last_motion_time = time.time()
        if not motion_detected:
            motion_detected = True
            write_log("INFO", "Motion detected — starting thermal camera", 1)
            # Start thermal thread
            if thermal_thread is None or not thermal_thread.is_alive():
                thermal_thread = ThermalCaptureThread( duration=duration,temp_min=temp_min,temp_max=temp_max,w=w,h=h,output_folder=output_folder)
                thermal_thread.start()
        # Save RGB snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(rgb_folder, f"rgb_{timestamp}.jpg")
        cv2.imwrite(filename, frame)        
        write_log(info="SAVE", message=f"Saved image: {filename}", verbose=1)
        time.sleep(1)
    elif motion_detected and (time.time() - last_motion_time > cooldown_seconds):
        motion_detected = False
        write_log("INFO", "Motion ended — stopping thermal camera", 1)
        if thermal_thread and thermal_thread.is_alive():
            thermal_thread.stop()
            thermal_thread.join()
            thermal_thread = None







