import FeedData as fd
import numpy as np
import JkBlades
import Vector


# Add "JackknifeTemplate" object with parameters "blades" and "sample"
# Add "JackknifeFeatures" with parameters "blades" and "trajectory"
# Terms:
# Trajectory is the incoming data stream from our camera feed

class Jackknife:
    def __init__(self, blades=JkBlades(), templates=fd.assemble_templates()):
        self.blades = blades
        self.templates = templates

    def classify(self, trajectory):
        features = JackknifeFeatures(self.blades, trajectory)
        template_cnt = self.templates.len()

        for tt, template, in enumerate(self.templates):
            cf = 1.0

            # Line 72


class JackknifeTemplate:
    def __init__(self, blades, sample):
        self.sample = sample
        self.gesture_id = sample.gesture_id
        self.lower = []
        self.upper = []

        self.lb - 1.0
        self.cf = 1.0

        self.rejection_threshold = np.inf()

        self.features = JackknifeFeatures(blades, self.trajectory)


class JackknifeFeatures:
    def __init__(self, blades=JkBlades, points=None):
        self.pts = blades
        self.pts = points
        self.pts = resample(self.pts, blades.resample_cnt)

        self.vecs = []
        self.abs = np.zeros(points.shape[1])
        minimum = Vector(self.pts[0])
        maximum = Vector(self.pts[0])

        for i in range(1, len(self.pts)):
            vec = self.pts[i] - self.pts[i - 1]

            self.abs += np.a


class Distributions:
    def __init__(self, max_score, bin_cnt):
        self.neg = Vector(0, bin_cnt)
        self.pos = Vector(0, bin_cnt)
        self.max_score = max_score
