import FeedData as fd

#Add "JackknifeTemplate" object with parameters "blades" and "sample"
#Add "JackknifeFeatures" with parameters "blades" and "trajectory"
#Terms:
    #Trajectory is the incoming data stream from our camera feed

class Jackknife:
    def __init__(self, blades, templates):
        self.blades = blades
        self.templates = fd.Assemble_Templates()
            

        def classify(self, trajectory):
            features = JackknifeFeatures(this.blades, trajectory)
            template_cnt = self.templates.len()

            for tt, template, in enumerate(self.templates):
                cf = 1.0


                #Line 72