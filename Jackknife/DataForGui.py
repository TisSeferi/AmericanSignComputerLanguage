import cv2
import mediapipe as mp
import numpy as np
import time
import os
import Jackknife as jk
from pathlib import Path
import threading
import collections as col
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import cv2
from PIL import Image, ImageTk
import builtins

X = 0
Y = 1

LIVE_BOOT_TIME = 0
PRE_RECORDED_BOOT_TIME = 1

NUM_POINTS = 21
DIMS = 2
NUM_POINTS_AND_DIMS = NUM_POINTS * DIMS

DEFAULT_TIMES = 'times.npy'

HAND_REF = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
    'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
    'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]
def capture_vid(video_name=1):
    spot = PRE_RECORDED_BOOT_TIME

    if isinstance(video_name, int):
        spot = LIVE_BOOT_TIME

    times = np.load(DEFAULT_TIMES)
    expected_boot_time = times[spot]

    print("Initializing Hand Detector")
    actual_boot_time = time.time()
    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        print("Error: Failed to open File.")
        exit()
    actual_boot_time = time.time() - actual_boot_time
    print("HandDetector initialized successfully.")

    times[spot] = expected_boot_time + actual_boot_time / 2
    np.save(DEFAULT_TIMES,times)

    return (cap, mp.solutions.hands.Hands())

def landmarks_to_frame(results):
    if results.multi_hand_landmarks:
        landmarks = [results.multi_hand_landmarks[0]]
        if landmarks:
            for handLms in landmarks:
                # Convert landmarks to dataframe
                points = handLms.landmark
                frame = np.zeros((NUM_POINTS, DIMS))
                for ii, lm in enumerate(points):
                    frame[ii][X] = lm.x
                    frame[ii][Y] = lm.y
        return frame 
    
class Settings:
    def __init__(self):
        #cv2 settings
        self.cv2_resize = (640, 480)
        
        #buffer settings
        self.buffer_window = 3
        self.buffer_fps = 30

        self.buffer_frames = self.buffer_window * self.buffer_fps
        self.buffer_length = self.buffer_frames * NUM_POINTS

        #templates
        self.template_raw_vids_folder = 'TestVideos/'
        self.test_raw_vids_folder = ''
        
        self.templates_data_folder = str(Path(__file__).resolve().parent.parent / 'templates')
        self.test_data_folder = ''

        #boot time
        self.boot_time = 5
        
        def recalculate():
            self.buffer_frames = self.buffer_window * self.buffer_fps
            self.buffer_length = self.buffer_frames * NUM_POINTS

class DataHandler:
    def __init__(self, settings=Settings()):
        self.settings = settings

        self.mp4 = None
        self.template = None

        self.templates = None
        self.recognizer = None

        self.classification = None
        self.recognizer = None
        self.train_jackknife()

    def update_templates(self):
        self.templates = []
        folder = self.settings.templates_data_folder
        for path in os.listdir(folder):
            name = path.split('.')[0]
            self.templates.append((name, np.load(folder + '/' + path)))
        
    def train_jackknife(self):
        self.update_templates()
        self.recognizer = jk.Jackknife(templates=self.templates)

    def process_video(self, video_name):
        cap, hands = capture_vid(video_name)

        data = []
        success, img = cap.read()

        while success:
            if not success:
                print("Error in reading!")
                cap.release()
                exit()

            # Image resizing for standardiztion
            img = cv2.resize(img, self.settings.cv2_resize)

            # Running recognition
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            # Extracting Landmarks
            if results.multi_hand_landmarks:
                frame = landmarks_to_frame(results)
                data.append(frame)

            success, img = cap.read()

        cap.release()
        cv2.destroyAllWindows()

        return data
    
    def save_as_template(self, data="", location = None, name=""):
        if location == -1:
            location = self.settings.templates_data_folder
        #Data can be an mp4 path string or a numpy array
        if isinstance(data, str):
            data = self.process_video(data)
            name = data.split('.')[0]
        elif name == "":
            print("Naming error, please include name when saving numpy array")

        np.save(location + "/" + name, data)
        print("Successfully saved: " + name)
        print("to " + location + "/" + name + ".npy")

    def save_as_test(self, data="", location = -1, name=""):
        if location == -1:
            location = self.settings.test_data_folder
        # Data:
        # String or numpy template
        if isinstance(data, str):
            data = self.process_video(data)
            name = data.split('.')[0]
        elif name == "":
            print("Naming error, please include name when saving numpy array")

        np.save(location + "/" + name, data)
        print("Successfully saved: " + name)
        print("to " + location + "/" + name + ".npy")

    def live_process(self):
        cap, hands = capture_vid()
        data = col.deque()
        success, img = cap.read()

        while success:
            if not success:
                print("Error in reading!")
                cap.release()
                exit()

            img = cv2.resize(img, self.settings.cv2_resize)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                frame = landmarks_to_frame(results)
                data.append(frame)
            
            if len(data) ==  - 1:
                data.popleft()
                trajectory = np.array(data.copy())
                t1 = threading.Thread(target = self.recognizer.classify, args = ((trajectory),))
                print(t1.start())
                
            success, img = cap.read()

        cap.release()
        cv2.destroyAllWindows()
    
    def classify(self, data):
        if isinstance(data, str):
            extension = data.split('.')[1]
            data = str(Path(__file__).resolve().parent.parent) + '\\' + data
            if extension == 'mp4':
                data = self.process_video(data)
            else:
                data = np.load(data)           

        self.recognizer.classify(data)

d = DataHandler()
#d.classify('test.mp4')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def append_to_console(message):
    def do_append():
        console_output.configure(state='normal')
        console_output.insert(tk.END, message + '\n')
        console_output.configure(state='disabled')
        console_output.see(tk.END)
    main.after(0, do_append)

def update_frame():
    ret, cv_image = cap.read()
    if ret:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = hands.process(cv_image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    cv_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        pil_image = Image.fromarray(cv_image)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        canvas.itemconfig(image_on_canvas, image=tk_image)
        canvas.image = tk_image

    main.after(10, update_frame)

def classify_video_thread(file_path):
    def run():
        append_to_console(f"Classifying data: {file_path}")
        result = d.classify(file_path) 
        append_to_console(f"Classification completed: {result}")
    threading.Thread(target=run).start()

def run_button_clicked():
    file_path = runP.get()
    if file_path:
        def run():
            original_print = builtins.print
            builtins.print = append_to_console 
            try:
                d.classify(file_path)
            finally:
                builtins.print = original_print
        threading.Thread(target=run).start()
    else:
        append_to_console("Please enter a file name or path.")

main = tk.Tk()
main.title('ASCL Prototype')
main.geometry('1200x700')

title_label = ttk.Label(main, text='ASCL', font='Calibri 24 bold')
title_label.pack(pady=10)

canvas = tk.Canvas(main, width=400, height=400)
canvas.pack(side=tk.LEFT, padx=20, pady=10)
image_on_canvas = canvas.create_image(200, 200, anchor='center')

record_frame = ttk.Frame(main)
record_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))

record_button = ttk.Button(record_frame, text='Record')
record_button.pack(side=tk.RIGHT, padx=10)

recordP = ttk.Entry(record_frame)
recordP.pack(side=tk.RIGHT, padx=10)

run_frame = ttk.Frame(main)
run_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))

run_button = ttk.Button(run_frame, text='Run', command=run_button_clicked)
run_button.pack(side=tk.RIGHT, padx=10)

runP = ttk.Entry(run_frame)
runP.pack(side=tk.RIGHT, padx=10)

live_frame = ttk.Frame(main)
live_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))

live_button = ttk.Button(live_frame, text='Live')
live_button.pack(side=tk.RIGHT, padx=10)

console_output = scrolledtext.ScrolledText(main, height=16)
console_output.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 20))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    append_to_console("Failed to open camera.")
    raise ValueError("Unable to open video source")

update_frame()

main.mainloop()

cap.release()
cv2.destroyAllWindows()






    










