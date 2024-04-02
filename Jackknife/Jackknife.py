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
        self.template_cnt = self.templates.len()

    def classify(self, trajectory):
        features = JackknifeFeatures(self.blades, trajectory)
        cf = 1.0

        for tt, template, in enumerate(self.templates):
            cf *= 1.0

            if self.blades.cf_abs_distance:
                cf *= 1.0 / max(
                    0.01, features.abs.dot(template.features.abs)
                )

            if self.blades.cf_bb_widths:
                cf *= 1.0 / max(
                    0.01, features.bb.dot(template.features.bb)
                )

            if self.blades.lower_bound:
                template.lb = cf * self.lower_bound(
                    features.vecs, template
                )

            #TODO sort templates ???


class JackknifeFeatures:
    def __init__(self, blades=JkBlades, points=None):
        self.blades = blades
        self.pts = points
        self.len = len(points)

        self.vecs = []
        self.abs = np.zeros(points.shape[1])




class Distributions:
    def __init__(self, max_score, bin_cnt):
        self.neg = Vector(0, bin_cnt)
        self.pos = Vector(0, bin_cnt)
        self.max_score = max_score
