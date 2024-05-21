import MacheteElement
import MacheteTrigger
import MVector as Vector
import MMAthematics as Mathematics
import ContinuousResult

class MacheteTemplate:
    def __init__(self, sample, device_id, cr_options, filtered=True):
        self.sample = sample
        self.points = []
        self.vectors = []
        self.device_id = device_id
        self.cr_options = cr_options 
        self.result = None          
        
        self.minimumFrameCount = 0
        self.maximumFrameCount = 0
        self.closedness = 0.0       
        self.f2l_Vector = Vector.Vector()      
        self.weightClosedness = 0.0 
        self.weightF2l = 0.0        
        self.vectorCount = 0        
        
        self.dtw = [[], []]         
        self.currentIndex = 0       
        self.sampleCount = 0        
        self.trigger = MacheteTrigger.MacheteTrigger() 

        resampled = []
        self.prepare(device_id, resampled, filtered)
        self.vectorCount = self.vectors.size()

        self.result = ContinuousResult(self.cr_options, self.sample.gesture_id, self.sample)

    def prepare(self, device_type, resampled, filtered=True):
        rotated = []
        dpPoints = []

        self.device_id = device_type

        if filtered:
            rotated = self.sample.filtered_trajectory
        else:
            rotated = self.sample.trajectory

        resampled.append(Vector(rotated[0]))

        rotated_size = rotated.size()
        for ii in range(1, rotated_size):
            count = resampled.size - 1
            length = resampled[count].l2norm(rotated[ii])

        #This is only referenced once on line 75 of MacheteTemplate???
            if length <= 1e-10:
                continue

            resampled.append(rotated[ii])
        
        sampleCount = resampled.size

        minimum, maximum = Mathematics.bounding_box(resampled)
        diag = maximum.l2norm(minimum)

        dp_points = Mathematics.douglas_peucker_density(resampled, diag * 0.010)

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
#
        #    ratio + v2.l2norm() / v1.l2norm()
#
        #    if ratio < 0.2:
        #        dpPoints.remove(len(dpPoints - 1))
        #        ptCnt = ptCnt - 1

        self.points = dpPoints

        self.vectors = Mathematics.vectorize(resampled, normalized=True)

        f2l_vector = self.points[self.points.size - 1] - self.points[0]
        f2l_length = f2l_vector.l2Norm()
        closedness = f2l_length
        closedness /= Mathematics.path_length(resampled)
        f2l_vector.Normalize()

        weightClosedness = (1.0 - f2l_length) / diag
        weightF2l = min(1.0, 2.0 * f2l_length / diag)


        #I NEED HELP WRITING LINES 141 THROUGH 182 in https://github.com/ISUE/Machete/blob/main/Assets/Scripts/Machete/MacheteTemplate.cs
        #I DO NOT KNOW HOW TO IMPLEMENT THE CONSTRUCTUR WITH OUR CURRENT LAYOUT


    #FINISHING THE REST OF THE FUNCTIONS TUESDAY
    def reset_elements(self):
        for ridx in range(0, 2):
            if self.dtw[ridx] is not None:
                self.dtw[ridx].clear()

        self.currentIndex

        #NOT SURE WHAT THIS IS?
        startAngleDegrees = 65.0

        #NOT SURE IF WE NEED THIS?
        #if (DeviceType.MOUSE == device_id)
        #   startAngleDegrees = 20.0

        for ridx in range(2):
            for cidx in range(self.vectorCount + 1):
                self.dtw[ridx].append(MacheteElement(cidx, startAngleDegrees))

        self.trigger.reset()

    def reset(self):
        self.reset_elements()

    def segmentation(self):
        current = self.dtw[self.currentIndex]
        curr = current[-1]

        #I CANT TELL WHERE THESE .Funcs ARE BEING CREATED?
        return (curr.startFrameNo - 1, curr.EndFrameNo)