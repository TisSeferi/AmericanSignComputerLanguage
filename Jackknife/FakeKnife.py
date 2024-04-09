import FeedData as fd
import numpy as np
import math
from JkBlades import JkBlades
from Vector import Vector
import JkTemplate
import mathematics


# Add "JackknifeTemplate" object with parameters "blades" and "sample"
# Add "JackknifeFeatures" with parameters "blades" and "trajectory"
# Terms:
# Trajectory is the incoming data stream from our camera feed

class Jackknife:
    def __init__(self, blades=JkBlades(), templates=fd.assemble_templates()):
        self.blades = blades
        self.templates = templates
    
    def add_template(self, sample):
        self.templates.append(JkTemplate(self.blades, sample))

    def classify(self, trajectory):

        features = JackknifeFeatures(self.blades, trajectory)
        template_cnt = self.templates.len

        for tt, in range(0, template_cnt):
            cf = 1.0

            if (self.blades.cf_abs_distance > 0):
                cf *= 1.0 / max(
                    0.01, features.abs.dot(self.template.features.abs)
                )

            if (self.blades.cf_bb_widths > 0):
                cf *= 1.0 / max(
                    0.01, features.bb.dot(self.template.features.bb)
                )
            
            self.templates[tt].cf = cf

            if (self.blades.lower_bound > 0):
                self.template.lb = cf * self.lower_bound(
                    features.vecs, self.template
                )

            #TODO sort templates ???
        self.templates.sort(self.compare_templates)
        best = np.inf
        ret = -1

        for tt in range(0, template_cnt):

            if (self.templates[tt].lb > self.templates[tt].rejection_threshold):
                continue
            if (self.templates[tt] > best):
                continue

            score = self.templates[tt].cf

            score *= self.DTW(features.vecs, self.templates[tt].features.vec)

            if (score > self.templates[tt].rejection_threshold):
                continue
            if (score < best):
                best = score
                ret = self.templates[tt].gesture_id
        
        return ret
    
    def train(self, gpsr_n, gpsr_r, beta):
        template_cnt = self.templates.len
        distributions = []
        synthetic = []

        worst_score = 0.0

        for ii in range(0, 1000):
            synthetic.len = 0

            for jj in range (0, 2):
                tt = math.floor(math.random() * template_cnt % template_cnt)

                s = self.templates[tt].sample
                len = s.trajectory.len

                start = math.floor(math.random() * (len / 2) % (len / 2))

                for kk in range(0, len):
                    synthetic.append(Vector(s.trajectory[start + kk]))

            features = JackknifeFeatures(self.blades, synthetic)

            for tt in range(0, template_cnt):
                score = self.DTW(features.vecs, self.templates[tt].features.vecs)

                if (worst_score < score):
                    worst_score = score
                if (ii > 50):
                    distributions[tt].add_negative_score(score)
            
            if (ii != 50):
                continue
            
            for tt in range(0, template_cnt):
                distributions.append(Distributions(worst_score, 1000))
            
        
        for tt in range(0, template_cnt):
            for ii in range(0, 1000):
                synthetic.len = 0

                ##TODO Write GPSR
                gpsr(self.templates[tt].sample.trajectory, synthetic, gpsr_n, 0.25, gpsr_r)

                features = JackknifeFeatures(self.blades, synthetic)
                score = self.DTW(features.vecs, self.templates[tt].features.vecs)
                distributions[tt].add_positive_score(score)

        for tt in range(0, template_cnt):
            threshold = distributions[tt].rejection_threshold(beta)
            self.templates[tt].rejection_threshold = threshold

    def DTW (self, v1, v2):
        cost = np.full(len(v1) + 1, len(v2) + 1, np.inf)
        cost[0, 0] = 0

        for ii in range(1, len(v1) + 1):
            start_j = max(1, ii - math.floor(self.blades.radius))
            end_j = min(len(v2), ii + math.floor(self.blades.radius))
            for jj in range(start_j, end_j + 1):
                
                cost[ii, jj] = dist + min(cost[ii - 1, jj], cost[ii, jj - 1], cost[ii - 1, jj - 1])

                if self.blades.inner_product:
                    dist = 1.0 - np.dot(v1[ii - 1], v2[jj - 1]) / (np.linalg.norm(v1[ii - 1]) * np.linalg.norm(v2[jj - 1]))
                elif self.blades.euclidean_dist:
                    dist = np.linalg.norm(v1[ii - 1] - v2[jj - 1])
                else:
                    assert(0)


        return cost[len(v1), len(v2)]
    
    def lower_bound(self, vecs, t):
        lb = 0.0
        component_cnt = vecs[0].data.len

        for ii in range(0, vecs.len):
            cost = 0.0

            for jj in range(0,component_cnt):
                if (self.blades.inner_product):
                    if (vecs[ii].data[jj] < 0.0):
                        cost += vecs[ii].data[jj] * t.lower[ii].data[jj]
                    else:
                        cost += vecs[ii].data[jj] * t.upper[ii].data[jj]

                elif (self.blades.euclidean_dist):
                    diff = 0.0

                    if (vecs[ii].data[jj] < t.lower[ii].data[jj]):
                        diff = vecs[ii].data[jj] - t.lower[ii].data[jj]
                    elif(vecs[ii].data[jj] > t.upper[ii].data[jj]):
                        diff = vecs[ii].data[jj] - t.upper[ii].data[jj]

                    cost += (diff * diff)
                else:
                    assert(0)

            if self.blades.inner_product:
                cost = 1.0 - min(1.0, max(-1.0, cost))
            
            lb += cost

        return lb

class JackknifeFeatures:
    def __init__(self, blades=JkBlades, points=None):
        self.pts = []
        self.vecs = []

        m = len(points[0].data)
        self.pts = mathematics.resample(points=points, n=blades.resample_cnt)

        #minimum = Vector(self.pts[0].data)
        minimum = self.pts[0]
        #maximum = Vector(self.pts[0].data)
        maximum = self.pts[0]

        #self.abs = Vector(0.0, m)

        self.abs = np.zeros(m)

        for ii in range(1, blades.resample_cnt):
            vec = self.pts[ii] - self.pts[ii - 1]
            #self.abs = np.absolute
            ##
            minimum = np.minimum(minimum, self.pts[ii])
            print(minimum)

            #for jj in range(m):
            #    self.abs.data[jj] += abs(vec.data[jj])
#
            #    #minimum.data[jj] = min(minimum.data[jj], self.pts[ii].data[jj])
            if (blades.inner_product):
                self.vecs.append(Vector(vec.normalize()))
            elif (blades.euclidean_distance):
                if (ii == 1):
                    self.vecs.append(Vector(self.pts[0]))

                    self.vecs.append(Vector(self.pts[ii]))
                else:
                    assert(0)

        if (blades.z_normalize):
            self.z_normalize(self.vecs)

        self.abs(self.normalize())
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
    
j = Jackknife()
data = np.load('test.npy')
print(j.classify(data))