import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import DataManagement as dm
import webbrowser
import PIL.ImageGrab
import keyboard
import pyautogui
import win32con
import win32api

class ASCL(self):
    def __init__(self, actions):
        self.initialize_template()


    def initialize_template(self, samplePts, theta, epsilon):
        self.template.buffer = []
        self.template.row = [[],[]]
        self.template.T = []
        self.template.startFrame = 0
        self.template.endFrame = 0
        self.template.currRowIdx = 0
        self.template.s1 = 0
        self.template.s2 = 0
        self.template.s3 = 0

        pts = angular_dp(samplePts, epsilon)
        N = 