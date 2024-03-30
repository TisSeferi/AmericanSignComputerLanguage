import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
USE_DATAFRAMES = False
NUM_POINTS = 21

BUFFER_WINDOW = 10 #In seconds
BUFFER_FPS = 30    #TODO Fix for variable framerate cameras
BUFFER_FRAMES = BUFFER_WINDOW * BUFFER_FPS

BUFFER_LENGTH = BUFFER_FRAMES * NUM_POINTS

RAW_VIDS_FOLDER = 'TestVideos'
TEMPLATES = 'templates'






HAND_REF = [
        'wrist',
        'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
        'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
        'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
        'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip',
        'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
    ]

def Euclidean_Dist(self, df1, df2, cols=['x', 'y']):
    return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)

def Process_Video(videoName):
    t = time.time()
    cap = cv2.VideoCapture(videoName)
    name = videoName.split('.')[0] + '-' + videoName.split('.')[1]
    name = name.split('/')[1]

    duration = cap.get(cv2.CAP_PROP_POS_MSEC)

    if not cap.isOpened():
        print("Error: Failed to open File.")
        exit()

    #print("HandDetector initialized successfully.")

    hands = mp.solutions.hands.Hands()
    
    #The list for returning the dataframes
    data =[]
    success, img = cap.read()
    while(success): 
        if not success:
            print("Error in reading!")
            cap.release()
            exit()

        #Image resizing for standardiztion
        img = cv2.resize(img, (640, 480))

        #Running recognition
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        #Extracting Landmarks
        if results.multi_hand_landmarks:
            landmarks = [results.multi_hand_landmarks[0]]
            if landmarks:
                for handLms in landmarks:
                    #Convert landmarks to dataframe
                    points = handLms.landmark
                    d = np.zeros((NUM_POINTS, 2))
                    for id, lm in enumerate(points):
                        d[id][0] = lm.x
                        d[id][1] = lm.y

                    if(USE_DATAFRAMES):
                        d = pd.DataFrame(data=d, columns=['x', 'y'], index=HAND_REF)

                    data.append(d)
        

        success, img = cap.read()

    cap.release()
    cv2.destroyAllWindows()


    if(USE_DATAFRAMES):
        data = pd.concat(data)
        data.to_csv('out.csv')
    else:        
        np.save(TEMPLATES + name, np.vstack(data))
    t = time.time() - t
    #print('Elapsed Time: ' + "%.2f" % t)
    return data

def Live_Process():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Failed to open File.")
        exit()

    print("HandDetector initialized successfully.")

    hands = mp.solutions.hands.Hands()
    
    #The list for returning the dataframes
    data = np.zeros((BUFFER_LENGTH, 2))
    frame = 0

    success, img = cap.read()
    while(success): 
        if not success:
            print("Error in reading!")
            cap.release()
            exit()

        #Image resizing for standardiztion
        img = cv2.resize(img, (640, 480))

        #Running recognition
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        #Extracting Landmarks
        if results.multi_hand_landmarks:
            landmarks = [results.multi_hand_landmarks[0]]
            if landmarks:
                for handLms in landmarks:
                    #Convert landmarks to dataframe
                    points = handLms.landmark
                    for id, lm in enumerate(points):
                        data[frame * NUM_POINTS + id][0] = lm.x
                        data[frame * NUM_POINTS + id][1] = lm.y                   
        else:
            for id in range(NUM_POINTS):
                data[frame * NUM_POINTS + id][0] = -1
                data[frame * NUM_POINTS + id][1] = -1

        frame = (frame + 1) % BUFFER_FRAMES
        print(data)
        success, img = cap.read()


    cap.release()
    cv2.destroyAllWindows()

# Web processing test
# Process_Video('https://www.pexels.com/download/video/3959694/')
    
def Extract_From_Vids():
    for str in os.listdir(RAW_VIDS_FOLDER):
        str = RAW_VIDS_FOLDER + str
        Process_Video(str)
        #TODO Move np.save here


def Assemble_Templates():
    list = []
    for str in os.listdir(TEMPLATES):
        list.append(np.load(TEMPLATES + '/' + str))

    return list

Live_Process()
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