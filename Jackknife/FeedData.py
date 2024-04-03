import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os


USE_DATAFRAMES = False
NUM_POINTS = 21

CV2_RESIZE = (640, 480)

BUFFER_WINDOW = 10  # In seconds
BUFFER_FPS = 30  # TODO Fix for variable framerate cameras
BUFFER_FRAMES = BUFFER_WINDOW * BUFFER_FPS

BUFFER_LENGTH = BUFFER_FRAMES * NUM_POINTS

RAW_VIDS_FOLDER = 'TestVideos/'
TEMPLATES = 'templates/'

HAND_REF = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
    'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
    'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]


def euclidean_dist(self, df1, df2, cols=['x', 'y']):
    return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)


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
    cap = cv2.VideoCapture(0)
    recognizer = jk()

    if not cap.isOpened():
        print("Error: Failed to open File.")
        exit()

    print("HandDetector initialized successfully.")

    hands = mp.solutions.hands.Hands()

    # The list for returning the dataframes
    data = np.zeros((BUFFER_LENGTH, 2))
    frame = 0

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
                    for ii, lm in enumerate(points):
                        data[frame * NUM_POINTS + ii][0] = lm.x
                        data[frame * NUM_POINTS + ii][1] = lm.y
        else:
            for ii in range(NUM_POINTS):
                data[frame * NUM_POINTS + ii][0] = -1
                data[frame * NUM_POINTS + ii][1] = -1

        frame = (frame + 1) % BUFFER_FRAMES
        recognizer.classify(data)

        success, img = cap.read()

    cap.release()
    cv2.destroyAllWindows()


# Web processing test
# Process_Video('https://www.pexels.com/download/video/3959694/')
def save_template(path):
    name = path.split('.')[0] + '-' + path.split('.')[1]
    path = RAW_VIDS_FOLDER + path
    data = process_video(path)
    np.save(TEMPLATES + name, data)
    print(data)


def extract_from_videos():
    for path in os.listdir(RAW_VIDS_FOLDER):
        save_template(path)


def assemble_templates():
    templates = []
    for path in os.listdir(TEMPLATES):
        templates.append(np.load(TEMPLATES + '/' + path))

    return templates


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
