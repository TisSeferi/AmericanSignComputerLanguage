from JkBlades import JkBlades
from JkFeatures import JkFeatures
import math
from Vector import Vector


class JkTemplate:
    #def __init__(self, blades=JkBlades(), sample=None):
    def __init__(self, blades=JkBlades(), sample=None, gid = None):
        self.sample = sample
        self.gesture_id = gid

        # TODO Identify gesture IDs

        self.lower = []
        self.upper = []

        self.lb = -1.0
        self.cf = 1.0

        self.rejection_threshold = float('inf')
        self.features = JkFeatures(blades, sample)

        vecs = self.features.vecs
        component_cnt = vecs[0].size()

        for ii in range(vecs.size()):
            maximum = Vector(component_cnt, float('inf') * -1)
            minimum = Vector(component_cnt, float('inf'))

            start = max(0, ii - math.floor(blades.radius))
            end = min(ii + blades.radius + 1, vecs.size())

            for jj in range(start, end):
                for kk in range(component_cnt):
                    maximum[kk] = max(
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