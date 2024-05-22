import MVector as Vector

class Mathematics:

    def bounding_box(trajectory):
        min_point = trajectory[0].clone()
        max_point = trajectory[0].clone()

        for ii in range(1, trajectory.size()):
            min_point.minimum(ii)
            max_point.maximum(ii)

        return min_point, max_point
    
    def douglas_peucker_r_density(self, points, splits, start, end, threshold):
        if (start + 1 > end):
            return
        
        AB = points[end] - points[start]
        denom = AB.dot(AB)

        largest = float('-inf')
        selected = -1

        for ii in range (start + 1, end):
            AC = points[ii] - points[start]
            numer = AC.dot(AB)
            d2 = AC.dot(AC) - numer * numer / denom

            if denom == 0.0:
                d2 = AC.l2norm()

            if d2 > largest:
                largest = d2
                selected = ii

        if largest < threshold:
            return
        
        self.douglas_peucker_r_density(points, start, selected, threshold, splits)
        self.douglas_peucker_r_density(points, selected, end, threshold, splits)

        splits[selected] = largest

    #def douglas_peucker_density(points, threshold):
    #    splits = [0] * 
    
    @staticmethod
    def vectorize(trajectory, normalize=True):
        vectors = []
        for ii in range(1, len(trajectory)):
            vec = trajectory[ii] - trajectory[ii - 1]
            if normalize:
                vec = vec.normalize()
            vectors.append(vec)
        return vectors