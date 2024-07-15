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

        #This is only referenced once on line 75 of MacheteTemplate???
            if length <= 1e-10:
                continue

            resampled.append(rotated[ii])
        
        self.sample_count = len(resampled)

        minimum, maximum = mathematics.bounding_box(resampled)
        diag = maximum.l2norm(minimum)

        dp_points = mathematics.douglas_peucker_density_trajectory(resampled, diag * 0.010)

    #We'll need to think on whether this implementation is necessary, do we WANT mouse type??
        #if (device_type == DeviceType.MOUSE):
        #    ptCnt = len(dp_points)
        #    v1 = dpPoints[1] - dpPoints[0]
        #    v2 = dpPoints[2] - dpPoints[1]
        #    ratio = v1.l2norm() / v2.l2norm()
#
        #    if ratio < 0.2:
        #        dpPoints.remove(0)
        #        ptCnt = ptCnt - 1
#
        #    v1 = dp_points[ptCnt - 2] - dp_points[ptCnt - 3]
        #    v2 = dp_points[ptCnt - 1] - dp_points[ptCnt - 2]
#````
        #    ratio + v2.l2norm() / v1.l2norm()
#
        #    if ratio < 0.2:
        #        dpPoints.remove(len(dpPoints - 1))
        #        ptCnt = ptCnt - 1

        #CPitt: DP returns a tuple and we want to grab the points, not the useless -inf data. 
        self.points = dp_points[1]

        self.vectors = mathematics.vectorize(resampled, normalize=True)

        f2l_vector = self.points[len(self.points) - 1] - self.points[0]
        f2l_length = f2l_vector.magnitude()
        self.closedness = f2l_length
        self.closedness /= mathematics.path_length(resampled)
        f2l_vector.normalize()

        self.weightClosedness = (1.0 - f2l_length) / diag
        self.weightF2l = min(1.0, 2.0 * f2l_length / diag)


    def reset_elements(self):
        for ridx in range(0, 2):
            if self.dtw[ridx] is not None:
                self.dtw[ridx].clear()

        self.current_index = 0

        start_angle_degrees = 20.0 if self.device_id == 'MOUSE' else 65.0

        #NOT SURE IF WE NEED THIS?
        #if (DeviceType.MOUSE == device_id)
        #   startAngleDegrees = 20.0

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

        self.current_index += self.current_index
        self.current_index %= 2

        current = self.dtw[self.current_index]

        current[0].start_frame_no = frame_no

        for col in range(1, self.vector_count):
            dot = nvec.dot(self.vectors[col - 1])
            cost = 1.0 - max(-1.0, min(1.0, dot))
            cost = cost ** 2

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

            #AGAIN NOT REALLY SURE IF WE NEED THIS SO I WILL LEAVE IT OUT FOR NOW

            #if (device_id == DeviceType.MOUSE)
            #{
            #    double cf_closedness = 2.0;
            #    double cf_f2l = 2.0;
#
            #    if (durationFrameCount < buffer.Count() - 1)
            #    {
            #        // Get first to last vector
            #        Vector vec = buffer[-1] - buffer[-(int) durationFrameCount];
            #        double total = current[vectorCount].total;
            #        double vlength = vec.L2Norm();
            #        double closedness1 = vlength / total;
#
            #        vec /= vlength;
            #        
            #        // Closedness
            #        cf_closedness = Math.Max(closedness1, closedness);
            #        cf_closedness /= Math.Min(closedness1, closedness);
            #        cf_closedness = 1 + weightClosedness * (cf_closedness - 1.0);
            #        cf_closedness = Math.Min(2.0, cf_closedness);
#
            #        // End direction
            #        double f2l_dot = f2l_Vector.Dot(vec);
            #        f2l_dot = 1.0 - Math.Max(-1.0, Math.Min(1.0, f2l_dot));
            #        cf_f2l = 1.0 + (f2l_dot / 2.0) * weightF2l;
            #        cf_f2l = Math.Min(2.0, cf_f2l);
#
            #        cf = cf_closedness * cf_f2l;
            #    }
            #}

            ret = curr.get_normalized_warping_path_cost()

            if duration_frame_count < self.minimum_frame_count:
                cf *= 1000

            self.trigger.update(frame_no, ret, cf, curr.start_frame_no, curr.end_frame_no)

            _t = self.trigger.get_threshold()

            self.result.update(ret * cf, _t, curr.start_frame_no, curr.end_frame_no, frame_no)

                