import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

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
        N = |pts|

        for i, elem, in enumerate(N):
            elem.startFrame = -1
            elem.endFrame = -1
            elem.cumulativeCost = 0
            elem.cumulativeLengt = 0
            elem.score = np.inf
            if (i == 0):
                elem.score = ((1-np.cos(np.linspace(0,2*np.pi,301)))**2)
            
            #Push-Back(template.row[0], elem)
            #Push-Back(template.row[1], elem)
                
            #Ask corey what this means (potentially is a template using the first and secoond row of matrix data)
                
            if (i > 0):
                vec1 = np.array(pts[i] - pts[i - 1])
                #Push-Back(template.T, )