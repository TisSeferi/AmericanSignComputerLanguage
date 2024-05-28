import math

class MacheteTrigger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = -1
        self.end = -1
        self.s1 = float('inf')
        self.s2 = float('inf')
        self.s3 = float('inf')
        self.minimum = False
        self.sum = 0.0
        self.count = 0.0

    def get_threshold(self):
        if self.count == 0:
            return float('inf')
        mu = self.sum / self.count
        return mu / 2.0
    
    def update(self, frame, score, cf, start, end):
        self.sum += score
        self.count += 1
        score *= cf

        self.s1 = self.s2
        self.s2 = self.s3
        self.s3 = score

        mu = self.sum / self.count if self.count != 0 else float('inf')
        threshold = mu / 2.0
        
        self.check = False

        if self.s3 < self.s2:
            self.start = start
            self.end = end
            return
        if self.s2 > threshold:
            return 
        if self.s1 < self.s2:
            return
        if self.s3 < self.s2:
            return
        
        self.check = True