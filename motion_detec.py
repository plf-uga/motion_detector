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


# RGB Camera Capture Thread
def capture_rgb(current_time):
    width, height = w, h
    ret, frame = cap.read()
    if not ret:
        write_log(info = "ERROR", message = "Error: Could not read RGB frame.", verbose = 1)
        return

    frame = cv2.resize(frame, (width, height))
   

    # Ensure "RGB" folder exists inside output_folder

    
    frame_path = os.path.join(rgb_folder, f"{current_time}.jpg")
    
    success = cv2.imwrite(frame_path, frame)  # Save frame
    if success:
        write_log(info = "INFO", message = f"Saved RGB frame: {frame_path}", verbose = 1)
    else:
        write_log(info = "ERROR", message = f"Error: Could not save RGB frame at {frame_path}", verbose = 1)


# Thermal Frame Callback
def py_frame_callback(frame, userptr):
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(array_pointer.contents, dtype=np.uint16).reshape(frame.contents.height, frame.contents.width)

    if not q.full():
        q.put(data)


##########################################################################
#---------------End of Util Functions -----------------------------------#
##########################################################################

# Open Arducam
index = find_Arducam()
cap = cv2.VideoCapture(index)
if not cap.isOpened():
    write_log(info = "ERROR", message = "Could not open RGB camera", verbose = 1)
    exit()


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


#PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)


#Initialize Background as float 
ret, background = cap.read()
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background_gray = cv2.resize(background_gray[:,:], (w, h))
background_float = background_gray.astype("float")

alpha = 0.05
trigger_threshold = 1500
x_size = 300
y_size = 100
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame[:,:],(w,h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.accumulateWeighted(gray, background_float, alpha)
    background_gray = cv2.convertScaleAbs(background_float)

    offset = gray.shape[0] //4 
    center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2 + offset
    top_left_x = center_x - x_size // 2
    top_left_y = center_y - y_size // 2
    bottom_right_x = center_x + x_size // 2
    bottom_right_y = center_y + y_size // 2
    
    
    # Compute absolute difference
    diff = cv2.absdiff(background_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Crop the central ROI box where cow is expected
    roi = thresh[center_y-y_size:center_y+y_size, center_x-x_size:center_x+x_size]
    motion_score = cv2.countNonZero(roi)
    #print(motion_score)

    # Save RGB image if motion is above threshold
    if motion_score > trigger_threshold:
        # Draw bounding box on the grayscale frame before saving
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back to BGR to draw in color
        #cv2.rectangle(
        ##    annotated,
        #    (top_left_x, top_left_y),
        #    (bottom_right_x, bottom_right_y),
        #    (0, 255, 0), 2
        #    )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(rgb_folder, f"rgb_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        write_log(info="SAVE", message=f"Saved image: {filename}", verbose=1)
        
        



        




