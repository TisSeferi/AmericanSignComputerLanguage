import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time


class HandDetector:
    def __init__(self):
        print("Initializing HandDetector...")
        self.cap = cv2.VideoCapture('TestVideo1.mp4')
        if not self.cap.isOpened():
            print("Error: Failed to open webcam.")
            exit()

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.arr = [0,0,0,0,0]
        self.test_flag = False
        self.scroll = False

        self.slow = 0

        print("HandDetector initialized successfully.")

        self.NUM_POINTS = 21
        self.HAND_REF = [
            'wrist',
            'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
            'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
            'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
        ]

    def get_frame_and_landmarks(self):
        success, img = self.cap.read()
        if not success:
            return None, None
        
        img = cv2.resize(img, (640, 480))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            hands_landmarks = [results.multi_hand_landmarks[0]]
            return img, hands_landmarks
        return img, None
    
    def draw_landmarks(self, img, hand_landmarks):
        if hand_landmarks:
            for handLms in hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def release(self):
        self.cap.release()

    def to_data_frame(self, landmark):
        d = np.zeros((self.NUM_POINTS, 3))
        for id, lm in enumerate(landmark):
            d[id][0] = lm.x
            d[id][1] = lm.y
            d[id][2] = lm.z

        df = pd.DataFrame(data=d, columns=['x', 'y', 'z'], index=self.HAND_REF)
        return df


    def Euclidean_Dist(self, df1, df2, cols=['x', 'y', 'z']):
        return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)


   
    def fings_up(self, df):
        tip_selection = [
            'thumb_tip',  # 4
            'index_finger_tip',  # 8,
            'middle_finger_tip',  # 12,
            'ring_finger_tip',  # 16,
            'pinky_tip',  # 18
        ]

        knuck_selection = [
            'thumb_mcp',  # 2,
            'index_finger_pip',  # 6,
            'middle_finger_pip',  # 10,
            'ring_finger_pip',  # 14,
            'pinky_pip',  # 18
        ]
        row_names = [
            'thumb',
            'index_finger',
            'middle_finger',
            'ring_finger',
            'pinky',
        ]
        cols = ['x','y','z']
        wrist = df.loc['wrist']
        thumb_ref = df.loc['index_finger_pip']

        knucks = self.Euclidean_Dist(df.loc[knuck_selection], wrist)
        knucks[0] = self.Euclidean_Dist(df.loc[knuck_selection], thumb_ref)[0] * .8 # thumb correction to index_finger_mcp
        tips = self.Euclidean_Dist(df.loc[tip_selection], wrist)
        tips[0] = self.Euclidean_Dist(df.loc[tip_selection], thumb_ref)[0]
        ret = np.greater_equal(tips, knucks)

        
        self.arr[0] += ret[0]
        self.arr[1] += ret[1]
        self.arr[2] += ret[2]
        self.arr[3] += ret[3]
        self.arr[4] += ret[4]       
    
        return tuple(ret)


hand_detector = HandDetector()
prevTime = 0

while True:
    img, landmarks = hand_detector.get_frame_and_landmarks()
    if img is None:
        continue

    img = hand_detector.draw_landmarks(img, landmarks)

    if landmarks:
        for handLms in landmarks:
            temp = hand_detector.to_data_frame(handLms.landmark)
            fingos = hand_detector.fings_up(temp)

            h, w, c = img.shape
            a = temp.loc['pinky_mcp']
            b = temp.loc['index_finger_mcp']
            ref = a if a['x'] > b['x'] else b

            x_offset = 20
            start_x = x_offset + int(ref['x'] * w)
            start_y = int(ref['y'] * h)

            end_x = start_x + 170
            dy = 30              

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Hand Gestures", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
        break

hand_detector.release()
cv2.destroyAllWindows()