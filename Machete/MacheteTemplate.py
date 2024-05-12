import MacheteElement
import MacheteTrigger
import MVector as Vector

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
            

