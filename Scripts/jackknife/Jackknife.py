import numpy as np
import math
import numba
from .JkBlades import JkBlades
from .JkTemplate import JkTemplate
from .JkFeatures import JkFeatures
import mathematics
import random as r


@numba.njit(cache=True)
def _dtw_dp(cost_mat, radius):
    """Sakoe-Chiba band DTW DP recurrence, JIT-compiled.

    cost_mat: (n1, n2) float64 C-contiguous array of pairwise costs.
    radius:   band half-width (int).
    Returns the scalar accumulated cost at dp[n1-1, n2-1].
    """
    n1, n2 = cost_mat.shape
    dp = np.full((n1 + 1, n2 + 1), np.inf)
    dp[0, 0] = 0.0

    for ii in range(1, n1 + 1):
        j_start = max(1, ii - radius)
        j_end = min(n2 + 1, ii + radius + 1)
        for jj in range(j_start, j_end):
            best = min(min(dp[ii - 1, jj], dp[ii, jj - 1]), dp[ii - 1, jj - 1])
            dp[ii, jj] = best + cost_mat[ii - 1, jj - 1]

    return dp[n1 - 1, n2 - 1]


GPSR_N = 6
GPSR_R = 2
BETA = 1.00
BINS = 1000
NUM_DIST_SAMPLES = 250


class Jackknife:
    def __init__(self, blades=JkBlades(), templates=None):
        self.blades = blades
        self.templates = []

        for name, trajectory in templates:
            self.add_template(sample=np.asarray(trajectory, dtype=np.float64), gid=name)

        self.length = len(self.templates)
        self.train(GPSR_N, GPSR_R, BETA)

    def add_template(self, sample, gid):
        self.templates.append(JkTemplate(self.blades, sample=sample, gid=gid))

    def is_match(self, trajectory, gid):
        features = JkFeatures(self.blades, trajectory)
        best_score = float('inf')
        ret = False

        for tid in range(len(self.templates)):

            if self.templates[tid].gesture_id != gid:
                continue

            cf = 1

            if self.blades.cf_abs_distance:
                cf *= 1.0 / max(0.01, features.abs.dot(self.templates[tid].features.abs))

            if self.blades.cf_bb_widths:
                cf *= 1.0 / max(0.01, features.bb.dot(self.templates[tid].features.bb))

            temp = self.templates[tid]
            temp.cf = cf
            self.templates[tid] = temp

            if self.blades.lower_bound:
                temp_lb = self.templates[tid]
                temp_lb.lb = cf * self.lower_bound(features.vecs, self.templates[tid])
                self.templates[tid] = temp_lb

            d = self.templates[tid].cf
            d *= self.DTW(features.vecs, self.templates[tid].features.vecs)

            if d < self.templates[tid].rejection_threshold:
                ret = True

            if d < best_score:
                best_score = d

        score = best_score
        return ret, score

    def classify(self, trajectory):
        trajectory = np.asarray(trajectory, dtype=np.float64)
        features = JkFeatures(self.blades, trajectory)
        template_cnt = len(self.templates)

        for tt in range(template_cnt):
            cf = 1.0

            if self.blades.cf_abs_distance > 0:
                cf *= 1.0 / max(0.01, features.abs.dot(self.templates[tt].features.abs))

            if self.blades.cf_bb_widths > 0:
                cf *= 1.0 / max(0.01, features.bb.dot(self.templates[tt].features.bb))

            self.templates[tt].cf = cf

            if self.blades.lower_bound > 0:
                self.templates[tt].lb = cf * self.lower_bound(features.vecs, self.templates[tt])

        self.templates = sorted(self.templates, key=lambda t: t.lb)
        best = float('inf')
        ret = -1

        for tt in range(template_cnt):

            if self.templates[tt].lb > self.templates[tt].rejection_threshold:
                continue
            if self.templates[tt].lb > best:
                continue

            score = self.templates[tt].cf
            score *= self.DTW(features.vecs, self.templates[tt].features.vecs)
            print(str(self.templates[tt].gesture_id) + " " + str(score))
            if score > self.templates[tt].rejection_threshold:
                continue
            if score < best:
                best = score
                ret = self.templates[tt].gesture_id

        print("1:" + str(ret))
        return (best, ret)

    def train(self, gpsr_n, gpsr_r, beta):
        template_cnt = len(self.templates)
        distributions = []

        worst_score = 0.0

        for ii in range(NUM_DIST_SAMPLES):
            pieces = []
            for jj in range(2):
                tt = int(r.random() * template_cnt) % template_cnt
                s = self.templates[tt].sample  # (n_frames, d) ndarray
                length = s.shape[0]
                half = max(1, int(length / 2))
                start = int(r.random() * half) % half
                pieces.append(s[start:start + half])

            synthetic = np.vstack(pieces)
            features = JkFeatures(self.blades, synthetic)

            for tt in range(template_cnt):
                score = self.DTW(features.vecs, self.templates[tt].features.vecs)
                if worst_score < score:
                    worst_score = score

                if ii > 50:
                    distributions[tt].add_negative_score(score)

            if ii != 50:
                continue

            for tt in range(template_cnt):
                distributions.append(Distributions(worst_score, BINS))

        for tt in range(template_cnt):
            for ii in range(NUM_DIST_SAMPLES):
                synthetic = mathematics.gpsr(self.templates[tt].sample, gpsr_n, 0.25, gpsr_r)
                features = JkFeatures(self.blades, synthetic)
                score = self.DTW(features.vecs, self.templates[tt].features.vecs)
                distributions[tt].add_positive_score(score)

        for tt in range(template_cnt):
            threshold = distributions[tt].rejection_threshold(beta)
            self.templates[tt].rejection_threshold = threshold

    def DTW(self, v1=None, v2=None):
        """Dynamic Time Warping with Sakoe-Chiba band.

        v1, v2: (n, d) ndarrays of feature vectors.
        Cost matrix is computed with numpy; the DP recurrence runs via numba njit.
        Returns scalar alignment cost.
        """
        if self.blades.inner_product and not self.blades.euclidean_distance:
            cost_mat = 1.0 - v1 @ v2.T  # (n1, n2)
        elif self.blades.euclidean_distance:
            cost_mat = (
                np.sum(v1 ** 2, axis=1, keepdims=True)
                + np.sum(v2 ** 2, axis=1)
                - 2.0 * v1 @ v2.T
            )  # (n1, n2)
        else:
            assert False, "Either inner_product or euclidean_distance must be set"

        radius = int(math.floor(self.blades.radius))
        return float(_dtw_dp(np.ascontiguousarray(cost_mat, dtype=np.float64), radius))

    def lower_bound(self, vecs, template):
        """Vectorized lower bound computation.

        vecs: (n_vecs, d) ndarray
        template.lower, template.upper: (n_vecs, d) ndarrays
        """
        if self.blades.inner_product:
            multiplier = np.where(vecs < 0, template.lower, template.upper)
            costs = np.sum(vecs * multiplier, axis=1)  # (n_vecs,)
            costs = 1.0 - np.clip(costs, -1.0, 1.0)
            return float(np.sum(costs))
        elif self.blades.euclidean_distance:
            diff = np.where(
                vecs < template.lower, vecs - template.lower,
                np.where(vecs > template.upper, vecs - template.upper, 0.0)
            )
            return float(np.sum(diff ** 2))
        else:
            raise ValueError("Invalid configuration for blades.")


class Distributions:
    def __init__(self, max_score, bin_cnt):
        self.neg = np.full(bin_cnt, 1e-8)
        self.pos = np.full(bin_cnt, 1e-8)
        self.max_score = max_score
        self.bin_cnt = bin_cnt

    def bin(self, score):
        pt1 = math.floor(score * (self.bin_cnt / self.max_score))
        return min(pt1, self.bin_cnt - 1)

    def add_negative_score(self, score):
        self.neg[self.bin(score)] += 1

    def add_positive_score(self, score):
        self.pos[self.bin(score)] += 1

    def rejection_threshold(self, beta):
        neg = self.neg / self.neg.sum()
        neg = np.cumsum(neg)
        assert abs(neg[-1] - 1.0) < 1e-4

        pos = self.pos / self.pos.sum()
        pos = np.cumsum(pos)
        assert abs(pos[-1] - 1.0) < 1e-4

        alpha = 1.0 / (1.0 + beta * beta)
        precision = pos / (pos + neg)
        recall = pos

        f_scores = 1.0 / (alpha / precision + (1.0 - alpha) / recall)
        best_idx = int(np.argmax(f_scores))

        ret = (best_idx + 0.5) * self.max_score / self.bin_cnt
        return ret
