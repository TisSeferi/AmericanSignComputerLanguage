import math


#Lets double check this constructor because I used a mix of stackoverflow and google so I'm not sure it's right?
class MacheteElement:
    def __init__(self, column, start_angle_degrees):
            
    #This is the constructor that is called when the column is updated
        self.column = column
        self.running_score = float('inf')
        self.total = 1e-10

        if self.column == 0:
            angle = math.radians(start_angle_degrees)
            threshold = 1.0 - math.cos(angle)
            self.score = threshold * threshold

            self.running_score = 0.0
            self.total = 0.0
        

    def get_normalized_warping_path_cost(self):
        if self.column == 0:
            return self.score
        return self.running_score / self.total

    def update(self, extend_this, frame_no, cost, length):
        self.start_frame_no = extend_this.start_frame_no
        self.end_frame_no = frame_no
        cost *= length

        self.running_score = extend_this.running_score + cost
        self.total = extend_this.total + length

#element = MacheteElement(0, 65.0)
#print(element.get_normalized_warping_path_cost())