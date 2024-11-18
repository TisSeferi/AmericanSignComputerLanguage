class ContinuousResultOptions:
    def __init__(self):
        self.latency_frame_count = 0
        self.individual_boundary = False
        self.abandon = False

class ResultState:
    WAIT_FOR_START = 0
    LOOKING_FOR_MINIMUM = 1
    TRIGGER = 2
    WAIT_FOR_END = 3

class ContinuousResult:
    def __init__(self, options, gid, sample):
        self.options = options
        self.gid = gid
        self.sample = sample
        self.reset()
        self.boundary = -1
        self.rejection_threshold = 0

    def triggered(self):
        return self.state == ResultState.TRIGGER
    
    def set_wait_for_start(self):
        self.state = ResultState.WAIT_FOR_START
        self.minimum = float('inf')
        self.score = self.minimum
        self.start_frame_no = -1
        self.end_frame_no = -1
    
    def reset(self):
        self.set_wait_for_start()
    
    def update(self, score, threshold, start_frame_no, end_frame_no, current_frame_no):
        
        if current_frame_no == -2:
            current_frame_no = end_frame_no

        if self.state == ResultState.WAIT_FOR_START:
            self.minimum = score
            if score < threshold:
                self.state = ResultState.LOOKING_FOR_MINIMUM

        if self.state == ResultState.LOOKING_FOR_MINIMUM:
            if score <= self.minimum:
                self.minimum = score
                self.score = self.minimum
                self.start_frame_no = start_frame_no
                self.end_frame_no = end_frame_no

            frame_cnt = current_frame_no - self.end_frame_no
            timeout = frame_cnt >= self.options.latency_frame_count
            if timeout:
                self.state = ResultState.TRIGGER
                return

        if self.state == ResultState.TRIGGER:
            self.state = ResultState.WAIT_FOR_END

        if self.state == ResultState.WAIT_FOR_END:
            advance = score > self.rejection_threshold
            if advance:
                self.state = ResultState.WAIT_FOR_START

    def set_wait_for_end(self, result):
        self.boundary = result.end_frame_no
        self.state = ResultState.WAIT_FOR_END

    def false_positive(self, result):
        self.reset()

    def state_str(self):
        if self.state == ResultState.LOOKING_FOR_MINIMUM:
            return "looking for minimum"
        elif self.state == ResultState.WAIT_FOR_START:
            return "wait for start"
        elif self.state == ResultState.TRIGGER:
            return "trigger"
        elif self.state == ResultState.WAIT_FOR_END:
            return "wait for end"
        else:
            return "impossible ResultState case"
    
    @staticmethod
    def select_result(results, cancel_with_something):
        triggered = []
        remaining = []

        for ii in range(0, len(results)):
            result = results[ii]

            if not result.triggered():
                continue
            
            triggered.append(result)
            # print(result)

        if len(triggered) == 0:
            return None
        
        for ii in range(0, len(triggered)):

            for jj in range(0, len(results)):
                result = results[jj]

                if triggered[ii] == result:
                    continue

                if triggered[ii].minimum > result.minimum:
                    if cancel_with_something:
                        triggered[ii].SET_WAIT_FOR_END(result)
                        break
            if triggered[ii].triggered():
                remaining.append(triggered[ii])

        if len(remaining) == 0:
            return None
        
        return remaining[0]
