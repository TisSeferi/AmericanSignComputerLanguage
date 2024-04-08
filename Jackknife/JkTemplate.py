import numpy as np
from JkBlades import JkBlades
from JkFeatures import JkFeatures
import mathematics
import Vector


class JkTemplate:
    def __init__(self, blades=JkBlades(), sample=None):
        self.sample = mathematics.flatten(sample)
        self.gesture_id = None
        self.features = JkFeatures(blades, sample)

        # TODO Identify gesture IDs

        self.lower = []
        self.upper = []

        self.lb = -1.0
        self.cf = -1.0

        self.rejection_threshold = np.inf

        vecs = self.features.vecs
        component_cnt = len(vecs[0])

        for ii, vec in enumerate(vecs):
            maximum = np.full((component_cnt, 2), np.inf)
            minimum = np.full((component_cnt, 2), -1 * np.inf)

            start = max(0, ii - int(blades.radius))
            end = min(ii + blades.radius + 1, len(vecs))

            for jj in range(start, end):
                for kk in range(component_cnt):
                    maximum[kk] = np.maximum(
                        maximum[kk],
                        vec[kk])

                    minimum[kk] = np.minimum(
                        minimum[kk],
                        vec[kk]
                    )
            self.upper.append(maximum)
            self.lower.append(minimum)

        self.lower = np.array(self.lower)
        self.upper = np.array(self.upper)

    def compare_templates(t1, t2):
        return t1.lb < t2.lb