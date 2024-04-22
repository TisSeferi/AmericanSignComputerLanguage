import cv2
import mediapipe as mp
import numpy as np
import time
import os
import Jackknife as jk
from pathlib import Path
import threading
import collections as col


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

            # Image resizing for standardiztion
            img = cv2.resize(img, self.settings.cv2_resize)

            # Running recognition
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            # Extracting Landmarks
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
            data = str(Path(__file__).resolve().parent.parent) + data
            if extension == 'mp4':
                data = self.process_video(data)
            else:
                data = np.load(data)           

        self.recognizer.classify(data)

d = DataHandler()
d.classify('test.mp4')



            













    










