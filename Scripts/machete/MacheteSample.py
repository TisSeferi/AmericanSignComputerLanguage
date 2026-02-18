import numpy as np


class Sample:
    def __init__(self, subject_id, gesture_id, instance_id):
        self.subject_id = subject_id
        self.gesture_id = gesture_id
        self.instance_id = instance_id
        self.gesture_name = ""
        self.sample_id = 0
        self.trajectory = np.empty((0, 0), dtype=np.float64)
        self.time_s = []

    def add_trajectory(self, data):
        self.trajectory = np.asarray(data, dtype=np.float64)

    def add_time_stamps(self, time_s):
        self.time_s = list(time_s)

    def clone(self):
        ret = Sample(self.subject_id, self.gesture_id, self.instance_id)
        ret.trajectory = np.copy(self.trajectory)
        ret.time_s = self.time_s.copy()
        return ret
