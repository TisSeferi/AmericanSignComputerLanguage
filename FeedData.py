import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import webbrowser
import PIL.ImageGrab
import keyboard
import pyautogui
import win32con
import win32api



class HandDetector:
    def __init__(self, actions):
        print("Initializing HandDetector...")
        self.cap = cv2.VideoCapture('TestVideo1.mp4')
        if not self.cap.isOpened():
            print("Error: Failed to open webcam.")
            exit()

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.actions = actions
        self.arr = [0,0,0,0,0]
        self.test_flag = False
        self.w, self.h = pyautogui.size()
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

        self.last_action = 0
        self.cooldown = 2

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

    def recognize_gesture(self, fingos):
        actionTime = time.time()
        if (actionTime - self.last_action > self.cooldown):
            action = gesture_actions.get(fingos)
            if action:
                action()
                self.last_action = actionTime


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

        if(ret[0] and ret[1] and ret[2] and ret[3] and ret[4]):
            self.test_flag = True

        if not (ret[0] or ret[1] or ret[2] or ret[3] or ret[4]):
            self.test_flag = False
            self.scroll = False

        if self.test_flag:
            self.slow += 1
            if self.slow == 2:
                self.slow = 0
                index_tip = df.loc['index_finger_mcp']
                x = int((self.w * 2 * (1 - index_tip['x']))) % self.w
                y = int((self.h * 1.1 * index_tip['y'])) % self.h

                win32api.SetCursorPos((x, y))
                

            if not (ret[1] and ret[2] and ret[3]):
                self.scroll = True


            if not ret[1]:
                pyautogui.click()
                

                
                

        
    
        return tuple(ret)

def launch_chrome():
    webbrowser.open('https://www.google.com')


def take_screenshot():
    im = PIL.ImageGrab.grab()
    im.show()

def alt_f4():
    keyboard.press('alt')
    keyboard.press('f4')
    keyboard.release('f4')
    keyboard.release('alt')

def close_window():
    keyboard.press('Ctrl')
    keyboard.press('w')
    keyboard.release('Ctrl')
    keyboard.release('w')

def set_flag(self):
    self.flag = True


gesture_actions = {
    (False, True, True, False, False): launch_chrome,
    (False, False, False, False, True): take_screenshot,
    (False, True, False, False, True): alt_f4,
    (False, True, True, True, False): close_window,

    }

hand_detector = HandDetector(gesture_actions)
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
            hand_detector.recognize_gesture(fingos)

            h, w, c = img.shape
            a = temp.loc['pinky_mcp']
            b = temp.loc['index_finger_mcp']
            ref = a if a['x'] > b['x'] else b

            x_offset = 20
            start_x = x_offset + int(ref['x'] * w)
            start_y = int(ref['y'] * h)

            end_x = start_x + 170
            dy = 30
            if fingos[2] and not(fingos[1] or fingos[3] or fingos[4]) :
                tip = (int(temp.loc['middle_finger_tip']['x'] * w), int(temp.loc['middle_finger_tip']['y'] * h))
                dip = (int(temp.loc['middle_finger_dip']['x'] * w), int(temp.loc['middle_finger_dip']['y'] * h))
                pip = (int(temp.loc['middle_finger_pip']['x'] * w), int(temp.loc['middle_finger_pip']['y'] * h))
                mcp = (int(temp.loc['middle_finger_mcp']['x'] * w), int(temp.loc['middle_finger_mcp']['y'] * h))

                #cv2.line(img, tip, dip, (0, 0, 0), 25)
                #cv2.line(img, dip, pip, (0, 0, 0), 25)
                #cv2.line(img, pip, mcp, (0, 0, 0), 25)

                text ='CENSORED'
                
                mdx = 50
                mdy = 20
                
                x, y = pip
                p0 = (x - mdx, y)
                p1 = (x + mdx , y + mdy)
                p2 = (x - mdx, y - mdy)

                cv2.rectangle(img, p1, p2, (0, 0, 0), -1)
                cv2.putText(img, text, p0, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                
           # cv2.rectangle(img, (start_x - 20, start_y - dy), (end_x, start_y + 5 * dy), (0, 0, 0), -1)
           # for ind, val in enumerate(['thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky']):
           #     text = f'{val}: {"Up" if fingos[ind] else "Down"}'
           #     cv2.putText(img, text, (start_x, start_y + ind * dy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Hand Gestures", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
        break

hand_detector.release()
cv2.destroyAllWindows()