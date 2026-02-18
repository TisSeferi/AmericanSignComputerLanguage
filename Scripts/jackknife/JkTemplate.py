import numpy as np
import math
from .JkBlades import JkBlades
from .JkFeatures import JkFeatures


class JkTemplate:
    def __init__(self, blades=JkBlades(), sample=None, gid=None):
        self.sample = sample  # (n_frames, d) ndarray
        self.gesture_id = gid

        self.lb = -1.0
        self.cf = 1.0
        self.rejection_threshold = float('inf')

        self.features = JkFeatures(blades, sample, is_template=True)

        vecs = self.features.vecs  # (n_vecs, d) ndarray
        n_vecs = len(vecs)

        lower_list = []
        upper_list = []

        for ii in range(n_vecs):
            start = max(0, ii - math.floor(blades.radius))
            end = min(n_vecs, ii + blades.radius + 1)
            window = vecs[start:end]  # (window_size, d)
            lower_list.append(window.min(0))
            upper_list.append(window.max(0))

        self.lower = np.array(lower_list)  # (n_vecs, d)
        self.upper = np.array(upper_list)  # (n_vecs, d)
