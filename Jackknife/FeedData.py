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

USE_DATAFRAMES = False
NUM_POINTS = 21
DIMS = 2
NUM_POINTS_AND_DIMS = NUM_POINTS * DIMS

CV2_RESIZE = (640, 480)

BUFFER_WINDOW = 3 # In seconds
BUFFER_FPS = 30  # TODO Fix for variable framerate cameras
BUFFER_FRAMES = BUFFER_WINDOW * BUFFER_FPS

BUFFER_LENGTH = BUFFER_FRAMES

RAW_VIDS_FOLDER = 'TestVideos/'
TEMPLATES = str(Path(__file__).resolve().parent.parent / 'templates')

HAND_REF = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
    'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
    'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]




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
        if results.multi_hand_landmarks:
            landmarks = [results.multi_hand_landmarks[0]]
            if landmarks:
                for handLms in landmarks:
                    # Convert landmarks to dataframe
                    points = handLms.landmark
                    d = np.zeros((NUM_POINTS, 2))
                    for id, lm in enumerate(points):
                        d[id][0] = lm.x
                        d[id][1] = lm.y

                    data.append(d)

        success, img = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    return data


def live_process():
    print("Starting hand detection")
    cap = cv2.VideoCapture(0)
    

    if not cap.isOpened():
        print("Error: Failed to open File.")
        exit()

    print("HandDetector initialized successfully.")

    hands = mp.solutions.hands.Hands()

    recognizer = jk.Jackknife(templates = assemble_templates())

    # The list for returning the dataframes
    #data = np.zeros((BUFFER_LENGTH, 2))
    data = col.deque()
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

        # Extracting Landmarks
        if results.multi_hand_landmarks:
            landmarks = [results.multi_hand_landmarks[0]]
            if landmarks:
                for handLms in landmarks:
                    # Convert landmarks to dataframe
                    points = handLms.landmark
                    for ii, lm in enumerate(points):
                        frame[ii][X] = lm.x
                        frame[ii][Y] = lm.y
            data.append(frame.copy())
        else:
            pass
            #for ii in range(NUM_POINTS):
            #    frame[ii][X] = -1
            #    frame[ii][Y] = -1

        
        if len(data) == BUFFER_FRAMES - 1:
            data.popleft()
            trajectory = np.array(data.copy())
            t1 = threading.Thread(target = recognizer.classify, args = ((trajectory),))
            print(t1.start())
            
        success, img = cap.read()

    cap.release()
    cv2.destroyAllWindows()


# Web processing test
# Process_Video('https://www.pexels.com/download/video/3959694/')
def save_template(path):
    name = path.split('.')[0]
    path = RAW_VIDS_FOLDER + path
    data = process_video(path)
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



#save_test('test.mp4')
#classify_example(process_video('test .mp4'))
#extract_from_videos()
live_process()



# print('Running Recognition')
# 
# t = time.time()
# 
# for str in os.listdir('TestVideos'):
#     str = 'TestVideos/' + str
#     Process_Video(str)
# 
# t = time.time() - t
# print('Elapsed Time: ' + "%.2f" % t)
#extract_from_videos()

#live_process()
#print(assemble_templates())
