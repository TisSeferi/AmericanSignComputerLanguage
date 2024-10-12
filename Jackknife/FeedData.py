import cv2
import mediapipe as mp
import numpy as np
import time
import os
import Jackknife as jk
from pathlib import Path
import threading
import collections as col
from Machete import Machete
from ContinuousResult import ContinuousResult, ContinuousResultOptions
from JackknifeConnector import JKConnector as jkc
from JkBlades import JkBlades
import cProfile

X = 0
Y = 1
Z = 2

USE_DATAFRAMES = False
NUM_POINTS = 21
DIMS = 3
NUM_POINTS_AND_DIMS = NUM_POINTS * DIMS

CV2_RESIZE = (640, 480)

BUFFER_WINDOW = 3  # In seconds
BUFFER_FPS = 15  # TODO Fix for variable framerate cameras
BUFFER_FRAMES = BUFFER_WINDOW * BUFFER_FPS

BUFFER_LENGTH = BUFFER_FRAMES

RAW_VIDS_FOLDER = 'templatevids/'

HOME = str(Path(__file__).resolve().parent.parent)
TEMPLATES = HOME + '\\templates\\'
TESTS = HOME + '\\testvideos\\'

HAND_REF = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
    'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
    'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]

class FrameCaptureThread(threading.Thread):
    def __init__(self, cap, buffer, frame_skip=2):
        super().__init__()
        self.cap = cap
        self.buffer = buffer
        self.frame_skip = frame_skip
        self.running = True
        self.frame_count = 0

    def run(self):
        while self.running:
            success, img = self.cap.read()
            if not success:
                break

            if self.frame_count % self.frame_skip == 0:
                self.buffer.append(img)

            self.frame_count += 1

    def stop(self):
        self.running = False
        self.join()

def landmarks_to_frame(results):
    if results.multi_hand_landmarks:
        landmarks = [results.multi_hand_landmarks[0]]
        if landmarks:
            for handLms in landmarks:
                # Convert landmarks to dataframe
                points = handLms.landmark
                frame = np.zeros((NUM_POINTS * DIMS))
                for ii, lm in enumerate(points):
                    ii = ii * 3
                    frame[ii + X] = lm.x
                    frame[ii + Y] = lm.y
                    frame[ii + Z] = lm.z
        return frame

def assemble_templates():
    templates = []
    for path in os.listdir(TEMPLATES):
        p = np.load(TEMPLATES + '/' + path)
        name = path.split('.')[0]
        templates.append((name, np.load(TEMPLATES + '/' + path)))

    return templates

def draw_landmarks(image, landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    if landmarks.multi_hand_landmarks:
        for hand_landmarks in landmarks.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def live_process():
    print("Starting hand detection")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Failed to open File.")
        exit()

    print("HandDetector initialized successfully.")

    cr = ContinuousResultOptions()
    hands = mp.solutions.hands.Hands()
    machete = Machete(device_type=None, cr_options=cr, templates=assemble_templates())
    print("Machete initialized successfully.")
    blades = JkBlades()
    blades.set_ip_defaults()
    blades.lower_bound = False
    blades.cf_abs_distance = False
    blades.cf_bb_widths = False
    recognizer_options = jk.Jackknife(templates=assemble_templates(), blades=blades)
    print("Recognizer initialized successfully.")

    frame_buffer = col.deque(maxlen=10)
    capture_thread = FrameCaptureThread(cap, frame_buffer)
    capture_thread.start()

    data = col.deque()
    ret = []
    current_count = 0
    hand_detected = False

    while True:
        if not frame_buffer:
            continue

        img = frame_buffer.popleft()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        draw_landmarks(img, results)

        if results.multi_hand_landmarks:
            hand_detected = True
            point = landmarks_to_frame(results)
            data.append(point)

            machete.process_frame(point, current_count, ret)

            result = ContinuousResult.select_result(ret, False)
        
            jk_buffer = jkc.get_jk_buffer_from_video(data, 0, current_count)

            if result is not None:
                match, recognizer_d = recognizer_options.is_match(trajectory=jk_buffer, gid=result.sample.gesture_id)
                if match:
                    print("Matched " + result.sample.gesture_id + " with score " + str(recognizer_d))

            current_count += 1
        else:
            if not hand_detected:
                print("Waiting for hand detection...")

        cv2.imshow("Hand Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture_thread.stop()
    cap.release()
    cv2.destroyAllWindows()

def machete_process(input):
    print("Starting hand detection")
    cap = cv2.VideoCapture(input)
    
    if not cap.isOpened():
        print("Error: Failed to open File.")
        exit()

    print("HandDetector initialized successfully.")

    cr = ContinuousResultOptions()
    hands = mp.solutions.hands.Hands()
    machete = Machete(device_type=None, cr_options=cr, templates=assemble_templates())
    print("Machete initialized successfully.")
    blades = JkBlades()
    blades.set_ip_defaults()
    blades.lower_bound = False
    blades.cf_abs_distance = False
    blades.cf_bb_widths = False
    recognizer_options = jk.Jackknife(templates=assemble_templates(), blades=blades)
    print("Recognizer initialized successfully.")

    data = col.deque()
    ret = []
    current_count = 0
    frame = np.zeros((NUM_POINTS, DIMS))

    success, img = cap.read()
    while success:
        if not success:
            print("Error in reading!")
            cap.release()
            exit()

        img = cv2.resize(img, CV2_RESIZE)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        point = landmarks_to_frame(results)
        data.append(point)

        machete.process_frame(point, current_count, ret)

        result = ContinuousResult.select_result(ret, False)
        
        jk_buffer = jkc.get_jk_buffer_from_video(data, 0, current_count)

        if result is not None:
            match, recognizer_d = recognizer_options.is_match(trajectory=jk_buffer, gid=result.sample.gesture_id)
            if match:
                print("Matched " + result.sample.gesture_id + " with score " + str(recognizer_d))

        current_count += 1

    cap.release()
    cv2.destroyAllWindows()

def machete_test(input="test.mp4"):
    machete_process(TESTS + input)
    
def process_video(video_name):
    t = time.time()
    cap = cv2.VideoCapture(video_name)
    duration = cap.get(cv2.CAP_PROP_POS_MSEC)

    if not cap.isOpened():
        print("Error: Failed to open File.")
        exit()

    hands = mp.solutions.hands.Hands()

    data = []
    success, img = cap.read()
    while success:
        if not success:
            print("Error in reading!")
            cap.release()
            exit()

        img = cv2.resize(img, CV2_RESIZE)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        data.append(landmarks_to_frame(results))

        success, img = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    return data


def save_template(path):
    name = path.split('.')[0]
    path = RAW_VIDS_FOLDER + path
    data = process_video(path)
    np.save(TEMPLATES + "/" + name, data)

def save_test(path):
    name = path.split('.')[0]
    data = process_video(path)
    np.save(name, data)

def extract_from_videos():
    for path in os.listdir(RAW_VIDS_FOLDER):
        save_template(path)

def classify_example(test):
    recognizer = jk.Jackknife(templates=assemble_templates())
    print(recognizer.classify(test))

def run_profile():
    live_process()

cProfile.run('run_profile()')

#extract_from_videos()
