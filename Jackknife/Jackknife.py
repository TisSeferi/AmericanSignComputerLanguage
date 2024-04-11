import FeedData as fd
from functools import cmp_to_key
import numpy as np
import math
from JkBlades import JkBlades
from Vector import Vector
from JkTemplate import JkTemplate
from JkTemplate import compare_templates
from JkFeatures import JkFeatures
import mathematics
import random as r

# Add "JackknifeTemplate" object with parameters "blades" and "sample"
# Add "JackknifeFeatures" with parameters "blades" and "trajectory"
# Terms:
# Trajectory is the incoming data stream from our camera feed
GPSR_N = 6
GPSR_R = 2
BETA = 1.00


class Jackknife:
    def __init__(self, blades=JkBlades(), templates=fd.assemble_templates()):
        self.blades = blades
        self.templates = Vector([])

        for ii, t in enumerate(templates):
            self.add_template(sample=Vector(t), gid=ii % 5)
        self.length = self.templates.size()
        self.train(GPSR_N, GPSR_R, BETA)

    def add_template(self, sample, gid):
        self.templates.append(JkTemplate(self.blades, sample=sample, gid=gid))

    def classify(self, trajectory):

        if isinstance(trajectory, self.sample):
            return self.classify(trajectory.trajectory)

        features = JkFeatures(self.blades, trajectory)
        template_cnt = len(self.templates)

        for tt in range(template_cnt):
            cf = 1.0

            if self.blades.cf_abs_distance > 0:
                cf *= 1.0 / max(
                    0.01, features.abs.dot(self.templates[tt].features.abs))

            if self.blades.cf_bb_widths > 0:
                cf *= 1.0 / max(
                    0.01, features.bb.dot(self.templates[tt].features.bb))

            self.templates[tt].cf = cf

            if self.blades.lower_bound > 0:
                self.templates[tt].lb = cf * self.lower_bound(features.vecs, self.templates[tt])
            # TODO sort templates ???

        self.templates = sorted(self.templates, key=cmp_to_key(compare_templates))
        best = np.inf
        ret = -1

        for tt in range(0, template_cnt):

            if self.templates[tt].lb > self.templates[tt].rejection_threshold:
                continue
            if self.templates[tt].lb > best:
                continue

            score = self.templates[tt].cf

            score *= self.DTW(features.vecs, self.templates[tt].features.vecs)

            if (score > self.templates[tt].rejection_threshold):
                continue
            if (score < best):
                best = score
                ret = self.templates[tt].gesture_id

        return ret

    def train(self, gpsr_n, gpsr_r, beta):
        template_cnt = self.templates.size()
        distributions = Vector([])
        synthetic = Vector([])

        worst_score = 0.0

        for ii in range(0, 50):
            synthetic.length = 0

            for jj in range(0, 2):
                tt = math.floor(r.random() * template_cnt % template_cnt)

                s = self.templates[tt].sample
                length = s.size()

                start = math.floor(r.random() * (length / 2) % (length / 2))

                for kk in range(0, int(length / 2)):
                    synthetic.append(Vector(s[start + kk]))

            features = JkFeatures(self.blades, synthetic)

            for tt in range(0, template_cnt):
                score = self.DTW(features.vecs, self.templates[tt].features.vecs)

                if worst_score < score:
                    worst_score = score

                if ii > 50:
                    distributions[tt].add_negative_score(score)

            if ii != 50:
                continue

            for tt in range(0, template_cnt):
                distributions.append(Distributions(worst_score, 1000))

        for tt in range(0, template_cnt):
            for ii in range(0, 50):
                synthetic = mathematics.gpsr(self.templates[tt].sample, synthetic, gpsr_n, 0.25, gpsr_r)
                print("124")
                print(np.shape(synthetic.data))
                features = JkFeatures(self.blades, synthetic)
                score = self.DTW(features.vecs, self.templates[tt].features.vecs)                
                distributions[tt].add_positive_score(score)

        for tt in range(0, template_cnt):
            threshold = distributions[tt].rejection_threshold(beta)
            self.templates[tt].rejection_threshold = threshold

    def DTW(self, v1=None, v2=None):
        cost = Vector([])

        for i in range(0, v1.size() + 1):
            cost.append(Vector(np.full(v2.size(), np.inf)))

        cost[0][0] = 0.0

        for ii in range(1, v1.size() + 1):
            for jj in range(max(1, ii - math.floor(self.blades.radius)),
                            min(v2.size(), ii + math.floor(self.blades.radius))):
                cost[ii][jj] = min(min(cost[ii - 1][jj], cost[ii][jj - 1]), cost[ii - 1][jj - 1])

            if self.blades.inner_product:
                cost[ii][jj] += 1.0 - v1[ii - 1].dot(v2[jj - 1])
            elif self.blades.euclidean_distance:
                cost[ii][jj] += v1[ii - 1].l2norm2(v2[jj - 1])
            else:
                assert 0

        # ls
        # "\n\nJK 153 [Score]:")
        # print(cost[v1.size() - 1][v2.size() - 1])
        return cost[v1.size() - 1][v2.size() - 1]

    def lower_bound(self, vecs, template):
        lb = 0.0
        component_cnt = vecs[0].data.length
        for ii, in range(len(vecs)):
            cost = 0.0

            for jj in range(component_cnt):
                if self.blades.inner_product:
                    if vecs[ii].data[jj] < 0.0:
                        cost += vecs[ii].data[jj] * template.lower[ii].data[jj]
                    else:
                        cost += vecs[ii].data[jj] * template.upper[ii].data[jj]
                elif self.blades.euclidean_distance:
                    diff = 0.0
                    if vecs[ii].data[jj] < template.lower[ii].data[jj]:
                        diff = vecs[ii].data[jj] - template.lower[ii].data[jj]
                    elif vecs[ii].data[jj] > template.upper[ii].data[jj]:
                        diff = vecs[ii].data[jj] - template.upper[ii].data[jj]
                    cost += diff ** 2
                else:
                    raise ValueError("Invalid configuration for blades.")

            if self.blades.inner_product:
                cost = 1.0 - min(1.0, max(-1.0, cost))

            lb += cost

        return lb


class Distributions:
    def __init__(self, max_score, bin_cnt):
        self.neg = Vector(0, bin_cnt)
        self.pos = Vector(0, bin_cnt)
        self.max_score = max_score

    def bin(self, score):
        pt1 = math.floor(score * (self.neg.size() / self.max_score))
        pt2 = self.neg.size() - 1
        return min(pt1, pt2)

    def add_negative_score(self, score):
        self.neg[self.bin(score)] += 1

    def add_positive_score(self, score):
        self.pos[self.bin(score)] += 1

    def rejection_threshold(self, beta):

        self.neg = self.neg.divide(self.neg.sum())
        self.neg = self.neg.cumulative_sum()
        assert (abs(self.neg.data[self.data.length - 1] - 1.0) < .00001)

        self.pos = self.pos.divide(self.pos.sum())
        self.pos = self.pos.cumulative_sum()
        assert (abs(self.pos.data[self.pos.data.length - 1] - 1.0) < .00001)

        alpha = 1.0 / (1.0 + beta * beta)
        precision = self.pos.divide((self.pos.add(self.neg)))

        recall = self.pos

        best_score = 0.0
        best_idx = -1

        for ii in range(0, len(self.neg)):
            # might need fixing
            E = (alpha / precision.data[ii]) + ((1.0 - alpha) / recall.data[ii])
            f_score = 1.0 / E

            if f_score > best_score:
                best_score = f_score
                best_idx = ii

        ret = best_idx + 0.5
        ret *= self.max_score / len(self.neg)

        return ret


j = Jackknife()
data = np.load('test.npy')
print(j.classify(data))
