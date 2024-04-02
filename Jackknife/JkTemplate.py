import numpy as np
import pyts
import JkBlades
import Jackknife as jk
import Vector


class JkTemplate:
    def __init__(self, blades=JkBlades(), sample=None):
        self.sample = sample
        self.gesture_id = None
        self.features = jk.JackknifeFeatures(blades, sample)

        # TODO Identify gesture IDs

        self.lower = []
        self.upper = []

        self.lb = -1.0
        self.cf = -1.0

        self.rejection_threshold = np.inf

        vecs = self.features.vecs
        component_cnt = vecs[0].data.length

        for ii, vec in enumerate(vecs.data):
            maximum = Vector(np.inf, component_cnt)
            minimum = Vector(-1 * np.inf, component_cnt)

            start = max(0, ii - int(blades.radius))
            end = min(ii + blades.radius + 1, vecs.length)

            for jj in range(start, end):
                for kk in range(component_cnt):
                    maximum.data[kk] = max(
                        maximum.data[kk],
                        vec.data[kk])

                    minimum.data[kk] = min(
                        minimum.data[kk],
                        vec.data[kk]
                    )
            self.upper.append(maximum)
            self.lower.append(minimum)

    def compare_templates(t1, t2):
        return t1.lb < t2.lb