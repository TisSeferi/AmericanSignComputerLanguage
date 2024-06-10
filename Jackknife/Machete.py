import math

from MVector import Vector
from MacheteTemplate import MacheteTemplate
from MacheteTrigger import MacheteTrigger
from CircularBuffer import CircularBuffer
from ContinuousResult import ContinuousResult
from MacheteSample import Sample
import numpy as np
import FeedData

class Machete:
    def __init__(self, device_type, cr_options):
        self.device_type = device_type
        self.cr_options = cr_options
        self.buffer = CircularBuffer()
        self.templates = []
        self.training_set = []
        self.last_frame_no = -1
        self.device_fps = -1
        self.best_score = float('inf')
        self.best_template = None
        self.last_pt = []

    #Hey Corey check this out
    def prepare(self):
        templates = FeedData.assemble_templates()
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
        samp = Sample(0,0,0)
        samp.add_trajectory(trajectory)
        self.add_sample(samp, filtered)


    def add_sample(self, sample, filtered):
        size = len(sample.trajectory) * 5
        if size > self.buffer.size():
            self.buffer.resize(size)

        template = MacheteTemplate(self.device_type, self.cr_options, sample, filtered)
        self.templates.append(template)
        self.training_set.append(sample)
        self.reset()
    
    def reset(self):
        for ii in range(0, len(self.templates)):
            self.templates[ii].reset()
        self.buffer.clear()

    def segmentation(self, score, head, tail):
        self.score = score
        self.head = head
        self.tail = tail

        score = self.best_score
        head = -1
        tail = -1

        if self.bestTemplate.trigger.check is True:
            head = self.bestTemplate.trigger.start
            tail = self.bestTemplate.trigger.end

    def process_frame(self, pt, frame_no, results):
        if self.last_frame_no == -1:
            self.last_pt = pt

        while self.last_frame_no < frame_no:
            self.buffer.insert(pt)
            self.last_frame_no += 1

        vec = Vector(pt - self.last_pt)
        segment_length = vec.l2norm()

        if self.device_type == 'MOUSE' and segment_length < 10.0:
            return

        self.last_pt = pt

        if segment_length <= 1e-10:
            return

        vec /= segment_length

        for ii in range(0, len(self.templates)):
            self.templates[ii].update(self.buffer, pt, vec, frame_no, segment_length)
            results.append(self.templates[ii].result)    

m = Machete(None, ContinuousResult)
samp = Sample(0,0,0)
samp.add_trajectory(np.load('tests/test.npy'))

m.add_sample(samp)
