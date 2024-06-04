
class ContinuousResultOptions:
    def __init__(self):
        self.latency_frame_count = 0
        self.individual_boundary = False
        self.abandon = False

class ContinuousResult:
    WAIT_FOR_START = 0
    LOOKING_FOR_MINIMUM = 1
    TRIGGER = 2
    WAIT_FOR_END = 3


    def __init__(self, options, gid, sample):
        self.options = options
        self.gid = gid
        self.sample = sample
        self.reset()
        self.boundary = -1

    def reset(self):
        self.state = ContinuousResult.WAIT_FOR_START
        self.minimum = float('inf')
        self.score = self.minimum
        self.start_frame_no = -1
        self.end_frame_no = -1

    def triggered(self):
        return self.state == ContinuousResult.TRIGGER
    
    def update(self, score, threshold, start_frame_no, end_frame_no, current_frame_no):
        if current_frame_no == -2:
            current_frame_no = end_frame_no

        if self.state == ContinuousResult.WAIT_FOR_START:
            self.minimum = score
            if score < threshold:
                self.state = ContinuousResult.LOOKING_FOR_MINIMUM

        if self.state == ContinuousResult.LOOKING_FOR_MINIMUM:
            if score <= self.minimum:
                self.minimum = score
                self.score = self.minimum
                self.start_frame_no = start_frame_no
                self.end_frame_no = end_frame_no

            timeout = (current_frame_no - self.start_frame_no) >= self.options.latency_frame_count
            if timeout:
                self.state = ContinuousResult.TRIGGER

        if self.state == ContinuousResult.TRIGGER:
            self.state = ContinuousResult.WAIT_FOR_END

        if self.state == ContinuousResult.WAIT_FOR_END:
            advance = score > threshold
            if advance:
                self.state = ContinuousResult.WAIT_FOR_START

    def state_str(self):
        state_strs = {
            ContinuousResult.LOOKING_FOR_MINIMUM: "looking for minimum",
            ContinuousResult.WAIT_FOR_START: "wait for start",
            ContinuousResult.TRIGGER: "trigger",
            ContinuousResult.WAIT_FOR_END: "wait for end"
        }

        return state_strs.get(self.state, "unknown state")