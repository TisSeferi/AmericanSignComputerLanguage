import numpy as np
from .MacheteTemplate import MacheteTemplate
from .MacheteTrigger import MacheteTrigger
from .CircularBuffer import CircularBuffer
from .ContinuousResult import ContinuousResult
from .MacheteSample import Sample


class Machete:
    def __init__(self, device_type, cr_options, templates):
        self.device_type = device_type
        self.cr_options = cr_options
        self.buffer = CircularBuffer()
        self.templates = []
        self.training_set = []
        self.last_frame_no = -1
        self.device_fps = -1

        for t in templates:
            self.add_array_sample(t)

    def get_training_set(self):
        return self.training_set

    def get_cr_options(self):
        return self.cr_options

    def clear(self):
        self.templates.clear()
        self.last_frame_no = -1

    def add_array_sample(self, trajectory, filtered=None):
        samp = Sample(0, trajectory[0], 0)
        samp.add_trajectory(trajectory[1])
        self.add_sample(samp, filtered)

    def add_sample(self, sample, filtered):
        size = len(sample.trajectory) * 5
        if size > self.buffer.size():
            self.buffer.resize(size)

        template = MacheteTemplate(sample=sample, device_id=self.device_type, cr_options=self.cr_options)
        self.templates.append(template)
        self.training_set.append(sample)
        self.reset()

    def reset(self):
        for ii in range(len(self.templates)):
            self.templates[ii].reset()
        self.buffer.clear()

    def segmentation(self, score, head, tail):
        self.score = score
        self.head = head
        self.tail = tail

        self.score = self.best_score
        self.head = -1
        self.tail = -1

        if self.bestTemplate.trigger.check is True:
            self.head = self.bestTemplate.trigger.start
            self.tail = self.bestTemplate.trigger.end

    def process_frame(self, pt, frame_no, results):
        if self.last_frame_no == -1:
            self.last_pt = pt

        while self.last_frame_no < frame_no:
            self.buffer.insert(pt)
            self.last_frame_no += 1

        delta = pt - self.last_pt
        segment_length = float(np.linalg.norm(delta))

        if self.device_type == 'MOUSE' and segment_length < 10.0:
            return

        self.last_pt = pt

        if segment_length <= 1e-10:
            return

        nvec = delta / segment_length

        for ii in range(len(self.templates)):
            if self.templates[ii].is_static:
                continue
            self.templates[ii].update(self.buffer, pt, nvec, frame_no, segment_length)
            results.append(self.templates[ii].result)
