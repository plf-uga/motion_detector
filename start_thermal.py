import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import configparser
import psutil
from PIL import Image, ImageTk

class ThermalCaptureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermal Motion Capture GUI")
        self.root.geometry("400x300")
        self.process = None

        # Set working directory and script path
        self.script_dir = "C:\\Users\\alves\\OneDrive\\Documents\\Anderson\\Python Scripts\\motion_detector"
        os.chdir(self.script_dir)

        # Load and display logo image
        try:
            logo_path = os.path.join(self.script_dir, "logo.png")
            image = Image.open(logo_path)
            image = image.resize((120, 120), Image.Resampling.LANCZOS)
            self.logo_img = ImageTk.PhotoImage(image)
            self.logo_label = tk.Label(root, image=self.logo_img)
            self.logo_label.pack(pady=5)
        except Exception as e:
            print(f"Logo could not be loaded: {e}")

        # CAM NAME input
        self.label = tk.Label(root, text="Enter Camera Name:")
        self.label.pack(pady=10)

        self.cam_name_entry = tk.Entry(root, width=30)
        self.cam_name_entry.pack()

        # Start Button
        self.start_button = tk.Button(
            root, text="Start Capture", command=self.start_capture,
            bg="green", fg="white", width=20
        )
        self.start_button.pack(pady=5)

        # Stop Button
        self.stop_button = tk.Button(
            root, text="Stop Capture", command=self.stop_capture,
            bg="red", fg="white", width=20
        )
        self.stop_button.pack(pady=5)

        # Status Label
        self.status_label = tk.Label(root, text="Status: Idle", fg="blue")
        self.status_label.pack(pady=10)

    def start_capture(self):
        cam_name = self.cam_name_entry.get().strip()
        if not cam_name:
            messagebox.showerror("Input Error", "Please enter a camera name.")
            return

        config_file = os.path.join(self.script_dir, "thermal.par")
        if not os.path.exists(config_file):
            messagebox.showerror("Missing File", f"Cannot find {config_file}")
            return

        try:
            # Update config
            config = configparser.ConfigParser()
            config.read(config_file)
            config.set("Camera", "cam_name", cam_name)

            with open(config_file, "w") as configfile:
                config.write(configfile)

            # Launch motion capture script
            self.process = subprocess.Popen(
                ["python", "motion_trigger_rgb.py"],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            self.status_label.config(text=f"Status: Running ({cam_name})", fg="green")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start process: {e}")
            self.status_label.config(text="Status: Error", fg="red")

    def stop_capture(self):
        if self.process:
            try:
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
                self.status_label.config(text="Status: Stopped", fg="red")
                self.process = None
            except Exception as e:
                messagebox.showerror("Error", f"Could not stop process: {e}")
        else:
            messagebox.showinfo("Info", "No running process to stop.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalCaptureGUI(root)
    root.mainloop()
