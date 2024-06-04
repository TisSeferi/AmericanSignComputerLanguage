from MVector import Vector


class Sample:
    def __init__(self, subject_id, gesture_id, instance_id):
        self.subject_id = subject_id
        self.gesture_id = gesture_id
        self.instance_id = instance_id
        self.gesture_name = ""
        self.sample_id = 0
        self.trajectory = []
        self.filtered_trajectory = []
        self.time_s = []

    def add_trajectory(self, trajectory):
        for i in range(0, len(trajectory)):
            self.trajectory.append(trajectory[i].clone())

    def add_time_stamps(self, time_s):
        for i in range (0, len(time_s)):
            self.time_s.append(time_s[i])

    def add_filtered_trajectory(self, filtered_trajectory):
        for i in range(0, len(filtered_trajectory)):
            self.filtered_trajectory.append(filtered_trajectory[i].clone())

    def clone(self):
        ret = Sample(self.subject_id, self.gesture_id, self.instance_id)
        ret.add_trajectory(self.trajectory)
        ret.add_time_stamps(self.time_s)
        ret.add_filtered_trajectory(self.filtered_trajectory)
        return ret