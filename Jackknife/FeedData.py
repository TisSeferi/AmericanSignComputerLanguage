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

X = 0
Y = 1
Z = 2

USE_DATAFRAMES = False
NUM_POINTS = 21
DIMS = 3
NUM_POINTS_AND_DIMS = NUM_POINTS * DIMS

CV2_RESIZE = (640, 480)

BUFFER_WINDOW = 3 # In seconds
BUFFER_FPS = 30  # TODO Fix for variable framerate cameras
BUFFER_FRAMES = BUFFER_WINDOW * BUFFER_FPS

BUFFER_LENGTH = BUFFER_FRAMES

RAW_VIDS_FOLDER = 'templatevids/'

HOME = str(Path(__file__).resolve().parent.parent)
TEMPLATES =  HOME + '\\templates\\'
TESTS = HOME + '\\testvideos\\'

HAND_REF = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
    'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
    'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]

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
                    frame[ii + Y]= lm.y
                    frame[ii + Z] = lm.z
        return frame 


def assemble_templates():
    templates = []
    for path in os.listdir(TEMPLATES):
        p = np.load(TEMPLATES + '/' + path)
        name = path.split('.')[0]
        templates.append((name, np.load(TEMPLATES + '/' + path)))

    
    return templates

#returns data
def process_video(video_name):
    t = time.time()
    cap = cv2.VideoCapture(video_name)
    duration = cap.get(cv2.CAP_PROP_POS_MSEC)

    if not cap.isOpened():
        print("Error: Failed to open File.")
        exit()

    # print("HandDetector initialized successfully.")

    hands = mp.solutions.hands.Hands()

    # The list for returning the dataframes
    data = []
    success, img = cap.read()
    while success:
        if not success:
            print("Error in reading!")
            cap.release()
            exit()

        # Image resizing for standardiztion
        img = cv2.resize(img, CV2_RESIZE)

        # Running recognition
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # Extracting Landmarks
        data.append(landmarks_to_frame(results))

        success, img = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    return data

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
    recognizer_options = jk.Jackknife(templates = assemble_templates(), blades = blades)
    print("Recognizer initialized successfully.")

    data = col.deque()
    ret = []
    current_count = 0
    frame = np.zeros((NUM_POINTS, DIMS))

    hand_detected = False

    while True:
        success, img = cap.read()
        if not success:
            print("Error in reading!")
            cap.release()
            exit()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        draw_landmarks(img, results)

        if results.multi_hand_landmarks:
            hand_detected = True  # Set flag to true when a hand is detected
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
                print("Waiting for hand detection...")  # Inform the user that the program is waiting for a hand
            # If no hand is detected and hand_detected is False, skip processing

        cv2.imshow("Hand Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
    recognizer_options = jk.Jackknife(templates = assemble_templates(), blades = blades)
    print("Recognizer initialized successfully.")
    # recognizer = jk.Jackknife(templates = assemble_templates())

    # The list for returning the dataframes
    #data = np.zeros((BUFFER_LENGTH, 2))
    data = col.deque()
    ret = []
    current_count = 0
    frame = np.zeros((NUM_POINTS, DIMS))

    success, img = cap.read()
    while success:
    # while frame < BUFFER_FRAMES - 1:
    # For running one recognition instance
        if not success:
            print("Error in reading!")
            cap.release()
            exit()

        # Image resizing for standardiztion
        img = cv2.resize(img, CV2_RESIZE)

        # Running recognition
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        point = landmarks_to_frame(results)
        # Extracting Landmarks
        data.append(point)

        machete.process_frame(point, current_count, ret)

        result = ContinuousResult.select_result(ret, False)
        
      
        jk_buffer = jkc.get_jk_buffer_from_video(data, 0, current_count)

        if result is not None:
            match, recognizer_d = recognizer_options.is_match(trajectory=jk_buffer, gid=result.sample.gesture_id)
            print("This is the match " + str(match) + " This is the gesture id " + result.sample.gesture_id + " This is the score " + str(recognizer_d))
            if match:
                print("Matched")
            #else:
                #print("Not Matched")

        current_count += 1

    cap.release()
    cv2.destroyAllWindows()

def machete_test(input = "test.mp4"):
    machete_process(TESTS + input)


# Web processing test
# Process_Video('https://www.pexels.com/download/video/3959694/')
def save_template(path):
    name = path.split('.')[0]
    path = RAW_VIDS_FOLDER + path
    data = process_video(path)
    print(data)
    np.save(TEMPLATES + "/" + name, data)
    print(TEMPLATES + "/" + name)
    # print(data)

def save_test(path):
    name = path.split('.')[0]
    data = process_video(path)
    np.save(name, data)

def extract_from_videos():
    for path in os.listdir(RAW_VIDS_FOLDER):
        save_template(path)

def classify_example(test):
    recognizer = jk.Jackknife(templates = assemble_templates())
    print(recognizer.classify(test))






#machete_test()
live_process()

