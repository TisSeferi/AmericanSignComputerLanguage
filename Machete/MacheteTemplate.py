import MacheteElement
import MacheteTrigger
import MVector as Vector
import MMAthematics as Mathematics

class MacheteTemplate:
    def __init__(self, sample, device_id):
        self.sample = sample
        self.points = Vector()
        self.vectors = Vector() 
        self.device_id = device_id
        
        self.minimumFrameCount = 0
        self.maximumFrameCount = 0
        
        self.closedness = 0.0       
        self.f2l_Vector = None      
        self.weightClosedness = 0.0 
        self.weightF2l = 0.0        
        
        self.vectorCount = 0        
        
        self.dtw = [[], []]         
        
        self.currentIndex = 0       
        self.sampleCount = 0        
        
        self.trigger = MacheteTrigger() 

        self.cr_options = None
        self.result = None  

    def prepare(self, device_type, resampled, filtered=True):
        rotated = Vector()
        dpPoints = []

        self.device_id = device_type

        if filtered:
            rotated = self.sample.filtered_trajectory
        else:
            rotated = self.sample.trajectory

        resampled.append(rotated[0])
        rotated_size = rotated.size
        for ii in range(1, rotated_size):
            count = resampled.size - 1
            length = count.l2norm(rotated[ii])

        #This is only referenced once on line 75 of MacheteTemplate???
            if length <= 1e-10:
                continue

            resampled.append(rotated[ii])
        
        sampleCount = resampled.size

        minimum, maximum = Mathematics.bounding_box(resampled)
        diag = maximum.l2norm(minimum)

        dp_points = self.douglas_peucker_density(resampled, diag * 0.010)

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

        points = dpPoints

        self.vectors = Mathematics.vectorize(resampled, normalized=True)

        f2l_vector = points[points.size - 1] - points[0]
        f2l_length = f2l_vector.l2Norm()
        closedness = f2l_length
        closedness /= Mathematics.PathLength(resampled)
        f2l_vector.Normalize()

        weightClosedness = (1.0 -f2l_length) / diag
        


