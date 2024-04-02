import FeedData as fd
import numpy as np
import math
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
        self.pts = []
        self.vecs = []

        m = len(points[0].data)
        resampled_pts = resample(points, self.pts, blades.resample_cnt)

        minimum = Vector(self.pts[0].data)
        maximum = Vector(self.pts[0].data)

        self.abs = Vector(0.0, m)

        for ii in range(1, len(blades.resample_cnt)):
            vec = self.pts[ii].subtract(self.pts[ii - 1])

            for jj in range(m):
                self.abs.data[jj] += abs(vec.data[jj])

                minimum.data[jj] = min(minimum.data[jj], self.pts[ii].data[jj])
            
            if (blades.inner_product):
                self.vecs.append(Vector(vec.normalize()))
            elif (blades.euclidean_distance):
                if (ii == 1):
                    self.vecs.append(Vector(self.pts[0]))

                    self.vecs.append(Vector(self.pts[ii]))
                else:
                    assert(0)

        if (blades.z_normalize):
            z_normalize(self.vecs)

        self.abs(normalize())
        self.bb = (maximum.subtract(minimum)).normalize()      






class Distributions:
    def __init__(self, max_score, bin_cnt):
        self.neg = Vector(0, bin_cnt)
        self.pos = Vector(0, bin_cnt)
        self.max_score = max_score

    def bin (self, score):
        return min(math.floor(score * (self.neg.len / self.max_score)), self.neg.len - 1)
    
    def add_negative_score(self, score):
        self.neg.data[self.bin(score)] += 1

    def add_positive_score(self, score):
        self.pos.data[self.bin(score)] += 1

    def rejection_threshold(self, beta):

        self.neg = self.neg.divide(self.neg.sum())
        self.neg.cumulative_sum()
        assert(abs(self.neg.data[self.neg.data.len - 1] - 1.0) < .00001)

        self.pos = self.pos.divide(self.pos.sum())
        self.pos.cumulative_sum()
        assert(abs(self.pos.data[self.pos.data.len - 1] - 1.0) < .00001)

        alpha = 1.0 / (1.0 + beta * beta)
        precision = self.pos.divide((self.pos.add(self.neg)))

        recall = self.pos

        best_score = 0.0
        best_idx = -1

        for ii in range(0, self.neg.len):
            
            E = (alpha / precision.data[ii]) + ((1.0 - alpha) / recall.data[ii])
            f_score = 1.0 / E

            if (f_score > best_score):
                best_score = f_score
                best_idx = ii

        ret = best_idx + 0.5
        ret *= self.max_score / self.neg.len
        
        return ret
    
    