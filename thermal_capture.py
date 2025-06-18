import os
import cv2
import time
import numpy as np
from datetime import datetime
from libuvc_wrapper import *
import configparser
from queue import Queue
from ctypes import *

BUF_SIZE = 2
q = Queue(BUF_SIZE)

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
    h = config.getint('Camera', 'img_height')
    w = config.getint('Camera', 'img_width')
    temp_max = config.getint('Temperature', 'temp_max')
    temp_min = config.getint('Temperature', 'temp_min')
    duration = config.getint('Time', 'duration')
    output = config.get('Output', 'folder')
    return cam_name, h, w, temp_max, temp_min, duration, output

def ktoc(val):
    return (val - 27315) / 100.0

def raw_to_colored(data, min_temp, max_temp):
    data_celsius = ktoc(data)
    data_normalized = np.clip((data_celsius - min_temp) / (max_temp - min_temp), 0, 1)
    data_normalized = (data_normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(data_normalized, cv2.COLORMAP_HOT)

def py_frame_callback(frame, _):
    frame_data = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(frame_data.contents, dtype=np.uint16).reshape((frame.contents.height, frame.contents.width))
    q.put(data)

def remove_lock():
    lock_file = "/tmp/thermal_capture.lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)
        
def capture_thermal():
    cam_name, h, w, temp_max, temp_min, duration, output = load_config()
    output_folder = os.path.join(output, f"{cam_name}_{datetime.now().strftime('%Y_%m_%d')}", "color_thermal")
    os.makedirs(output_folder, exist_ok=True)

    ctx, dev, devh, ctrl = POINTER(uvc_context)(), POINTER(uvc_device)(), POINTER(uvc_device_handle)(), uvc_stream_ctrl()
    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0:
        write_log("ERROR", "uvc_init failed")
        return

    try:
        res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
        if res < 0:
            write_log("ERROR", "uvc_find_device failed")
            return

        res = libuvc.uvc_open(dev, byref(devh))
        if res < 0:
            write_log("ERROR", "uvc_open failed")
            return

        formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
        libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
                                               formats[0].wWidth, formats[0].wHeight,
                                               int(1e7 / formats[0].dwDefaultFrameInterval))

        cb = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)
        libuvc.uvc_start_streaming(devh, byref(ctrl), cb, None, 0)
        write_log("INFO", "Thermal camera started")

        start = time.time()
        while time.time() - start < duration:
            try:
                data = q.get(timeout=1)
                data_resized = cv2.resize(data, (w, h))
                colored_img = raw_to_colored(data_resized, temp_min, temp_max)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(output_folder, f"thermal_{timestamp}.jpg")
                cv2.imwrite(filename, colored_img)
                write_log("SAVE", f"Saved thermal image: {filename}")
                time.sleep(1)
            except:
                continue

        libuvc.uvc_stop_streaming(devh)
        write_log("INFO", "Thermal camera stopped")
    finally:
        libuvc.uvc_exit(ctx)

if __name__ == "__main__":
    try:
        capture_thermal()
    finally:
        remove_lock()
