import FeedData as fd
import numpy as np
import math
from JkBlades import JkBlades
from Vector import Vector
from JkTemplate import JkTemplate
from JkFeatures import JkFeatures
import mathematics


# Add "JackknifeTemplate" object with parameters "blades" and "sample"
# Add "JackknifeFeatures" with parameters "blades" and "trajectory"
# Terms:
# Trajectory is the incoming data stream from our camera feed

class Jackknife:
    def __init__(self, blades=JkBlades(), templates=fd.assemble_templates()):
        self.blades = blades
        self.templates = []
        for t in templates:
            self.add_template(t)
    
    def add_template(self, sample):
        self.templates.append(JkTemplate(self.blades, sample))

    def classify(self, trajectory):

        features = JkFeatures(self.blades, trajectory)
        template_cnt = int(len(self.templates))

        for t in self.templates:
            cf = 1.0

            if self.blades.cf_abs_distance:
                print("\n\n\n\n\n\n\n")
                print(features.abs)
                print("------")
                print(t.features.abs)
                print("\n\n\n\n\n\n\n")
                cf *= 1.0 / max(
                    0.01, np.dot(features.abs, t.features.abs)
                )

            if self.blades.cf_bb_widths:
                cf *= 1.0 / max(
                    0.01, np.dot(features.bb, t.features.bb)
                )
            
            t.cf = cf

            if self.blades.lower_bound:
                t.lb = cf * self.lower_bound(
                    features.vecs, t
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

                start = int(math.random() * (len / 2) % (len / 2))

                for kk in range(0, len):
                    synthetic.append(Vector(s.trajectory[start + kk]))

            features = JkFeatures(self.blades, synthetic)

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
                mathematics.gpsr(self.templates[tt].sample.trajectory, synthetic, gpsr_n, 0.25, gpsr_r)

                features = JkFeatures(self.blades, synthetic)
                score = self.DTW(features.vecs, self.templates[tt].features.vecs)
                distributions[tt].add_positive_score(score)

        for tt in range(0, template_cnt):
            threshold = distributions[tt].rejection_threshold(beta)
            self.templates[tt].rejection_threshold = threshold

    def DTW (self, v1, v2):
        cost = np.full(len(v1) + 1, len(v2) + 1, np.inf)
        cost[0, 0] = 0

        for ii in range(1, len(v1) + 1):
            start_j = max(1, ii - int(self.blades.radius))
            end_j = min(len(v2), ii + int(self.blades.radius))
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
                        cost += vecs[ii][jj] * t.lower[ii][jj]
                    else:
                        cost += vecs[ii][jj] * t.upper[ii][jj]

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