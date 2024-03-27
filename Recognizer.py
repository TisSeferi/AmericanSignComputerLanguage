import mediapipe as mp
import numpy as np
import pandas as pd
import math
import FeedData

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

        pts = self.angular_dp(samplePts, epsilon)
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
        diagLength = diagonal_length(pts)
        length = path_length(pts)

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
        
        self.angular_dp_recursive(trajectory, 0, N - 1, newPts, epsilon)
        newPts.append(trajectory[N - 1])

        return newPts
    
    def angular_dp_recursive(self, trajectory, start, end, newPts, epsilon):
        
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
        
        self.angular_dp_recursive(trajectory, start, selected, newPts, epsilon)
        newPts.append(trajectory[selected])
        self.angular_dp_recursive(trajectory, selected, end, newPts, epsilon)

    def calculate_correction_factors(self, template, cdpElem):

        f2l = template.buffer[cdpElem.endFrame] - template.buffer[cdpElem.startFrame]
        f2lLength = np.linalg.norm(f2l)
        openness = f2lLength / cdpElem.cumulativeLength

        cfopenness = 1 + template.Wclosedness * ((max(openness, template.openness), min(openness, template.openness)) - 1)
        cfopenness = min(2, cfopenness)

        cff2l = 1 + 1/2 * template.wf2l * (1 - np.dot(f2l/f2lLength, template.f2l))
        cff2l = min(2, cff2l)

        return (cfopenness * cff2l)
    
    def consume_input (self, template, x, frameNumber):

        template.buffer.append(x)

        length = np.linalg.norm(x - template.prev)
        if (length < 0):
            return
        
        x = (x - template.prev) / length
        template.prev = x

        prevRow = template.row[template.currRowIdx]
        template.currRowIdx = (template.currRowIdx + 1) % 2
        currRow = template.row[template.currRowIdx]
        currRow[0].startFrame = frameNumber

        T = template.T
        TN = abs(T)

        for col in range(1, TN):
            best = currRow[col - 1]
            path2 = prevRow[col - 1]
            path3 = prevRow[col]

            if (path2.score <= best.score):
                best = path2
            if (path3.score <= best.score):
                best = path3

            localCost = length * ((1 - np.dot(x, t[col - 1]))**2)
            currRow[col].startFrame = best.startFrame
            currRow[col].endFrame = frameNumber
            currRow[col].cumulativeCost = best.cumulativeCost + localCost
            currRow[col].cumulativeLength = best.cumulativeLength + length
            currRow[col].score = currRow[col].cumulativeCost / currRow[col].cumulativeLength

        cf = self.calculate_correction_factors(template, currRow[TN])
        correctedScore = cf * currRow[TN].score

        template.doCheck = False
        template.total = template.total + currRow[TN].score
        template.n = template.n + 1
        template.s1 = template.s2
        template.s3 = correctedScore

        if (template.s3 < template.s2):
            template.startFrame = currRow[TN].startFrame
            template.endFrame = currRow[TN].endFrame
            return
        
        mu = template.total / (2 * template.n)
        template.doCheck = (template.s2 < mu and template.s2 < template.s1 and template.s2 < template.s3)