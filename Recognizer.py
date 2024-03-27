import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import math

class Machete:
    def __init__(self, theta, epsilon):
        self.theta = np.radians(theta)
        self.epsilon = epsilon


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

        pts = self.angular_DP(samplePts, epsilon)
        N = len(pts)

        for i, elem, in enumerate(N):
            elem.startFrame = -1
            elem.endFrame = -1
            elem.cumulativeCost = 0
            elem.cumulativeLengt = 0
            elem.score = np.inf
            if (i == 0):
                elem.score = ((1-np.cos(np.linspace(0,2*np.pi,301)))**2)
            
            self.template.row[0].append(elem)
            self.template.row[1].append(elem)
 
            if (i > 0):
                vec = pts[i] - pts[i - 1]
                self.template.T.append(vec / np.linalg.norm(vec))
        
        f2l = pts[N - 1] - pts[0]
        diagLength = Diagonal_Length(pts)
        length = path_Length(pts)

        self.template.f2l = f2l / np.linalg.norm(f2l)
        self.template.openness = np.linalg.norm(f2l) / length
        self.template.Wclosedness = (1 - (np.linalg.norm(f2l) / diagLength))
        self.template.Wf2l = min(1, 2*(np.linalg.norm(f2l) / diagLength))

        return self.template
    
    ##john smells
    def angular_dp(self, trajectory, epsilon):
        diagLength = diagonal_length(trajectory)
        epsilon = diagLength * epsilon

        newPts = {}
        N = abs(trajectory)
        newPts.append(trajectory[0])
        
        self.Angular_DP_Recursive(trajectory, 0, N - 1, newPts, epsilon)
        newPts.append(trajectory[N - 1])

        return newPts
    
    def Angular_DP_Recursive(self, trajectory, start, end, newPts, epsilon):
        
        if (start + 1 >= end):
            return
            
        AB = trajectory[end] - trajectory[start]
        denom = np.dot(AB, AB)

        if (denom == 0):
            return
        
        largest = epsilon
        selected = -1

        for idx in range(start + 1, end):
            AC = trajectory[idx] - trajectory[start]
            numer = np.dot(AB, AC)
            d2 = np.dot(AC, AC) - ((numer**2) / denom)

            vec1 = trajectory[idx] - trajectory[start]
            vec2 = trajectory[end] - trajectory[idx]
            l1 = np.linalg.norm(vec1)
            l2 = np.linalg.norm(vec2)

            if (l1 * l2 == 0):
                continue

            d = (np.dot(vec1, vec2) / (l1 * l2))
            distance = ((d2 * math.acos(d)) / math.pi)

            if (distance >= largest):
                largest = distance
                selected = idx
        
        if (selected == -1):
            return
        
        self.Angular_DP_Recursive(trajectory, start, selected, newPts, epsilon)
        newPts.append(trajectory[selected])
        self.Angular_DP_Recursive(trajectory, selected, end, newPts, epsilon)

    def calculate_correction_factors(template, cdpElem):

        f2l = template.buffer[cdpElem.endFrame] - template.buffer[cdpElem.startFrame]
        f2lLength = np.linalg.norm(f2l)
        openness = f2lLength / cdpElem.cumulativeLength

        cfopenness = 1 + template.Wclosedness * ((max(openness, template.openness), min(openness, template.openness)) - 1)
        cfopenness = min(2, cfopenness)

        cff2l = 1 + 1/2 * template.wf2l * (1 - np.dot(f2l/f2lLength, template.f2l))
        cff2l = min(2, cff2l)

        return (cfopenness * cff2l)
        return (cfopenness * cff2l)
