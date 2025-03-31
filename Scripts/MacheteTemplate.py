from MacheteElement import MacheteElement
from MacheteTrigger import MacheteTrigger
from Vector import Vector
import mathematics
from ContinuousResult import ContinuousResult

class MacheteTemplate:
    def __init__(self, sample, device_id, cr_options, filtered=True):
        self.sample = sample
        self.points = []
        self.vectors = []
        self.device_id = device_id
        self.cr_options = cr_options 
        self.result = None      

        self.minimum_frame_count = 0
        self.maximum_frame_count = 0
        self.closedness = 0.0       
        self.f2l_vector = []      
        self.weight_closedness = 0.0 
        self.weight_f2l = 0.0        
        self.vector_count = 0       
        
        self.dtw = [[], []]         
        self.current_index = 0       
        self.sample_count = 0        
        self.trigger = MacheteTrigger() 

        resampled = []
        self.prepare(device_id, resampled, filtered)
        self.vector_count = len(self.vectors)


        self.result = ContinuousResult(cr_options, sample.gesture_id, sample)


    def prepare(self, device_type, resampled, filtered=True):
        rotated = self.sample.filtered_trajectory if filtered else self.sample.trajectory

        resampled.append(rotated[0])
        self.device_id = device_type

        for ii in range(1, len(rotated)):
            count = len(resampled) - 1
            length = resampled[count].l2norm(rotated[ii])

            if length <= 1e-10:
                continue

            resampled.append(rotated[ii])
        
        self.sample_count = len(resampled)

        minimum, maximum = mathematics.bounding_box(resampled)
        diag = maximum.l2norm(minimum)

        dp_points = mathematics.douglas_peucker_density_trajectory(resampled, diag * 0.010)

        #CPitt: DP returns a tuple and we want to grab the points, not the useless -inf data. 
        self.points = dp_points[1]

        self.vectors = mathematics.vectorize(resampled, normalize=True)

        f2l_vector = self.points[len(self.points) - 1] - self.points[0]
        f2l_length = f2l_vector.magnitude()
        path_length = mathematics.path_length(resampled)
        self.closedness = f2l_length
        self.closedness /= path_length
        f2l_vector = f2l_vector.normalize()

        self.is_static = True if path_length / diag < 1.2 else False

        self.weightClosedness = (1.0 - f2l_length) / diag
        self.weightF2l = min(1.0, 2.0 * f2l_length / diag)


    def reset_elements(self):
        for ridx in range(0, 2):
            if self.dtw[ridx] is not None:
                self.dtw[ridx].clear()

        self.current_index = 0

        start_angle_degrees = 20.0 if self.device_id == 'MOUSE' else 65.0

        for ridx in range(2):
            for cidx in range(self.vector_count + 1):
                self.dtw[ridx].append(MacheteElement(cidx, start_angle_degrees))

        self.trigger.reset()

    def reset(self):
        self.reset_elements()

    def segmentation(self):
        current = self.dtw[self.current_index]
        curr = current[-1]

        return (curr.start_frame_no - 1, curr.end_frame_no)
    
    #Consume-Input(template, x, frameNumber)
    def update(self, buffer, pt, nvec, frame_no, length):
        previous = self.dtw[self.current_index]

        self.current_index += 1
        self.current_index %= 2

        current = self.dtw[self.current_index]

        current[0].start_frame_no = frame_no

        for col in range(1, self.vector_count + 1):
            dot = nvec.dot(self.vectors[col - 1])
            cost = 1.0 - max(-1.0, min(1.0, dot))
            cost = cost * cost

            n1 = current[col - 1]
            n2 = previous[col - 1]
            n3 = previous[col]

            extend = n1
            minimum = n1.get_normalized_warping_path_cost()

            if n2.get_normalized_warping_path_cost() < minimum:
                extend = n2
                minimum = n2.get_normalized_warping_path_cost()

            if n3.get_normalized_warping_path_cost() < minimum:
                extend = n3
                minimum = n3.get_normalized_warping_path_cost()

            current[col].update(extend, frame_no, cost, length)

        curr = current[self.vector_count]

        start_frame_no = curr.start_frame_no
        end_frame_no = curr.end_frame_no
        duration_frame_count = end_frame_no - start_frame_no + 1
        cf = 1.0

        ret = curr.get_normalized_warping_path_cost()

        if duration_frame_count < self.minimum_frame_count:
            cf *= 1000

        self.trigger.update(frame_no, ret, cf, curr.start_frame_no, curr.end_frame_no)

        _t = self.trigger.get_threshold()
        self.result.update(ret * cf, _t, curr.start_frame_no, curr.end_frame_no, frame_no)

                