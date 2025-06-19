import os
import cv2
import time
import subprocess
from datetime import datetime
import configparser

#####################################################################################
######################---Define Util Functions ---###################################
####################################################################################


LOCK_FILE = "thermal_capture.lock"
python_path = "/usr/bin/python3"
script_path = "temp_capture.py"

os.chdir("/home/alveslab/motion_detector")

def is_thermal_running():
    return os.path.exists(LOCK_FILE)

def create_lock():
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

def remove_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

def write_log(info, message, verbose):
    current_timestamp = time.time()   
    # Convert the timestamp to the desired format
    current_time = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(current_timestamp))
    cout = f'{info} {current_time} : {message}  \n'
    with open('thermal_outputs.log', 'a') as file:
        if verbose == 1:
            print(cout)
        file.write(cout)  


# Configuration
def load_config():
    config = configparser.ConfigParser()
    config.read('thermal.par')
    cam_name = config.get('Camera', 'cam_name')
    h = config.getint('Camera', 'img_height')
    w = config.getint('Camera', 'img_width')    
    temp_max = config.getint('Temperature', 'temp_max')
    temp_min = config.getint('Temperature','temp_min')
    thresh = config.getint('Temperature','temp_trigger')
    duration = config.getint('Time','duration')
    length = config.getint('Time', 'trigger_duration') 
    output = config.get('Output', 'folder')
    return cam_name, h, w, temp_max, temp_min, thresh, duration, length, output


def find_Arducam():
    #index = int(0)
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
                    if w > 160 and h > 120 and c ==3: 
                        index = i
                        print(f"Saving Arducam as index {i}!")
                else:
                   print(f"[WARN] Camera index {i} exists but could not be open")
                cap.release()
    return int(index)


class UserClosedProgramError(Exception):
    """Custom exception raised when the user explicitly closes the program."""
    def __init__(self, message="Program closed by user action."):
        self.message = message
        super().__init__(self.message)

##########################################################################
#---------------End of Util Functions -----------------------------------#
##########################################################################

# Open Arducam
ind = find_Arducam()
cap = cv2.VideoCapture(ind)
if not cap.isOpened():
    write_log(info = "ERROR", message = "Could not open RGB camera", verbose = 1)
    exit()


#Import program parameters
cam_name, h, w, temp_max, temp_min, thresh, duration, length, output = load_config()

#cam_name, motion_thresh, duration, output = load_config()
output_folder = os.path.join(output, f"{cam_name}_{datetime.now().strftime('%Y_%m_%d')}")
rgb_folder = os.path.join(output_folder, "RGB")
os.makedirs(rgb_folder, exist_ok=True)

# Ensure output folder exists
output_folder = os.path.join(output, f"{cam_name}_{datetime.now().strftime('%Y_%m_%d')}")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "color_thermal"), exist_ok = True)
os.makedirs(os.path.join(output_folder, "raw_thermal"), exist_ok = True)
rgb_folder = os.path.join(output_folder, "RGB")
os.makedirs(rgb_folder, exist_ok=True)  # <-- FIX
write_log(info = "INFO", message = f"Created directory: {output_folder}", verbose = 1)


##########################################################################
#---------------Begin of Main Program -----------------------------------#
##########################################################################


#Initialize Background as float 
ret, background = cap.read()
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background_gray = cv2.resize(background_gray[:,:], (w, h))
background_float = background_gray.astype("float")

trigger_threshold = 5000
x_size = 300
y_size = 100
alpha = 0.05


start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if (time.time() - start_time) > length*60:
                    #print(time.time() - start_time)
                    write_log(info = "INFO", message ="RGB-trigger program completed.", verbose = 1)
                    remove_lock()
                    cap.release()
                    break
    try:
        frame = cv2.resize(frame[:,:], (w, h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        #print(gray.shape)
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
  

        if motion_score > trigger_threshold: # and (time.time() - trigger_time > cooldown_seconds):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rgb_filename = os.path.join(rgb_folder, f"rgb_{timestamp}.jpg")
            cv2.imwrite(rgb_filename, frame)
            write_log("SAVE", f"RGB image saved: {rgb_filename}", verbose = 1)

            if not is_thermal_running():
                write_log("INFO", "Motion detected — launching thermal capture script", verbose = 1)
                create_lock()
                subprocess.Popen([python_path, 'temp_capture.py'])
            #else:
            #    write_log("INFO", "Motion detected — but thermal capture already running", verbose = 1)

            #last_trigger_time = time.time()

    except UserClosedProgramError as e:
        print(f"Caught custom exception: {e}")
        cap.release()
        remove_lock()
        sys.exit(0) # Exit cleanly after handling the custom exception
    except Exception as e:
        cap.release()
        remove_lock()
        print(f"An unexpected error occurred: {e}")
        sys.exit(1) # Exit with an error code for other exceptions
    #finally:
    #    cap.release()
    #    remove_lock()

time.sleep(0.1)
