import mediapipe as mp
import numpy as np
import math
import Jackknife.FeedData as FeedData
from MVector import Vector
import MacheteTemplate

class Machete:
    def __init__(self, theta, epsilon):
        self.theta = np.radians(theta)
        self.epsilon = epsilon


    def initialize_template(self, samplePts, theta, epsilon):
        self.template = MacheteTemplate()
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
            elem.score = float('inf')
            if (i == 0):
                elem.score = ((1 - math.cos(theta)) ** 2)
            
            self.template.row[0].append(elem)
            self.template.row[1].append(elem)
 
            if (i > 0):
                vec = Vector(pts[i]) - Vector(pts[i - 1])
                normalized_vec = vec.normalize()
                self.template.T.append(normalized_vec)
        
        f2l = Vector(pts[N - 1] - pts[0])
        diagLength = self.diagonal_length(pts)
        length = self.path_length(pts)

        self.template.f2l = f2l
        norm_f2l = self.normalize_vector(f2l)
        self.template.openness = math.sqrt(sum(x ** 2 for x in norm_f2l)) / self.path_length
        self.template.wclosedness = 1 - math.sqrt(sum(x ** 2 for x in norm_f2l)) / self.diagonal_length
        self.template.wf2l = min(1, 2 * math.sqrt(sum(x ** 2 for x in norm_f2l)) / self.diagonal_length)

        return self.template
    
    def diagonal_length(pts):
        
        mins = [min(coord) for coord in zip(*pts)]
        maxs = [max(coord) for coord in zip(*pts)]
        
        return math.sqrt(sum((maxs[i] - mins[i]) ** 2 for i in range(len(mins))))

    
    def path_length(pts):
       
        return sum(math.sqrt(sum((pts[i][j] - pts[i-1][j])**2 for j in range(len(pts[i])))) for i in range(1, len(pts)))

    def angular_dp(self, trajectory, epsilon):
        diagLength = self.diagonal_length(trajectory)
        epsilon = diagLength * epsilon

        newPts = []
        N = len(trajectory)
        newPts.append(trajectory[0])
        
        self.angular_dp_recursive(trajectory, 0, N - 1, newPts, epsilon)
        newPts.append(trajectory[N - 1])

        return newPts
    
    def angular_dp_recursive(self, trajectory, start, end, newPts, epsilon):
        
        if (start + 1 >= end):
            return
            
        AB = Vector(trajectory[end] - trajectory[start])
        denom = AB.dot()

        if (denom == 0):
            return
        
        largest = epsilon
        selected = -1

        for idx in range(start + 1, end - 1):
            AC = Vector(trajectory[idx] - trajectory[start])
            numer = AB.dot(AC)
            d2 = AC.dot() - ((numer**2) / denom)

            vec1 = Vector(trajectory[idx] - trajectory[start])
            vec2 = Vector(trajectory[end] - trajectory[idx])
            l1 = vec1.normalize()
            l2 = vec2.normalize()

            if (l1 * l2 == 0):
                continue

            d = vec1.dot(vec2) / (l1 * l2)
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

        f2l = Vector(template.buffer[cdpElem.endFrame] - template.buffer[cdpElem.startFrame])
        f2lLength = f2l.normalize()
        openness = f2lLength / cdpElem.cumulativeLength

        cfopenness = 1 + template.Wclosedness * ((max(openness, template.openness), min(openness, template.openness)) - 1)
        cfopenness = min(2, cfopenness)

        cff2l = 1 + 1/2 * template.wf2l * (1 - f2l.dot(template.f2l))
        cff2l = min(2, cff2l)

        return (cfopenness * cff2l)
    
    def consume_input (self, template, x, frameNumber):

        template.buffer.append(x)
        diffx = Vector(x - template.prev)
        length = diffx.normalize()
        if (length < 0):
            return
        
        x = Vector((x - template.prev) / length)
        template.prev = x

        prevRow = template.row[template.currRowIdx]
        template.currRowIdx = (template.currRowIdx + 1) % 2
        currRow = template.row[template.currRowIdx]
        currRow[0].startFrame = frameNumber

        T = Vector(template.T)
        TN = T.normalize()
        t = 0

        for col in range(1, TN):
            best = currRow[col - 1]
            path2 = prevRow[col - 1]
            path3 = prevRow[col]

            if (path2.score <= best.score):
                best = path2
            if (path3.score <= best.score):
                best = path3

            localCost = length * ((1 - x.dot(T[col - 1]))**2)
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
