

#Add "JackknifeTemplate" object with parameters "blades" and "sample"
#Add "JackknifeFeatures" with parameters "blades" and "trajectory"
#add "blades" property "cf_abs_distance"

class Jackknife:
    def __init__(self, blades, templates):
        self.blades = blades
        self.templates = []

        def add_template(self, sample):
            self.templates.append(JackknifeTemplate(self.blades, sample))

        def classify(self, trajectory):
            features = JackknifeFeatures(this.blades, trajectory)
            template_cnt = self.templates.len()

            for tt, template, in enumerate(self.templates):
                cf = 1.0


                #Line 72