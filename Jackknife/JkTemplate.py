import numpy as np
from JkBlades import JkBlades
from JkFeatures import JkFeatures
import math
import Vector


class JkTemplate:
    #def __init__(self, blades=JkBlades(), sample=None):
    def __init__(self, blades=JkBlades(), sample=None, gid = None):
        #self.gesture_id = None
        self.gesture_id = gid

        # TODO Identify gesture IDs

        self.lower = []
        self.upper = []

        self.lb = -1.0
        self.cf = -1.0

        self.rejection_threshold = np.inf
        self.features = JkFeatures(blades, sample)

        vecs = self.features.vecs
        component_cnt = len(vecs[0].data)

        for ii in range(len(vecs)):
            maximum = Vector(np.inf * -1, component_cnt)
            minimum = Vector(np.inf, component_cnt)

            start = max(0, ii - math.floor(blades.radius))
            end = min(ii + blades.radius + 1, len(vecs))

            for jj in range(start, end):
                for kk in range(component_cnt):
                    maximum[kk] = np.maximum(
                        maximum.data[kk],
                        vecs[jj].data[kk])

                    minimum.data[kk] = min(
                        minimum.data[kk],
                        vecs[jj].data[kk]
                    )
            self.upper.append(maximum)
            self.lower.append(minimum)

def compare_templates(t1, t2):
    return t1.lb < t2.lb