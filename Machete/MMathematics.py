from MVector import Vector
import math

class Mathematics:

    def bounding_box(trajectory):
        min_point = trajectory[0].clone()
        max_point = trajectory[0].clone()

        for ii in range(1, len(trajectory)):
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

            v1 = points[ii] - points[start]
            v2 = points[end] - points[ii]

            l1 = v1.l2norm()
            l2 = v2.l2norm()

            self.dot = v1.dot(v2)
            self.dot /= (l1 * l2 > 0) if (l1 * l2) else 1.0
            self.dot = max(-1.0, min(1.0, self.dot))
            angle = math.acos(self.dot)
            d2 *= angle / math.pi

            if d2 > largest:
                largest = d2
                selected = ii

        if selected == -1:
            return
        
        largest = max(0.0, largest)
        largest = math.sqrt(largest)

        if largest < threshold:
            return

        Mathematics.douglas_peucker_r_density(points, splits, start, selected, threshold)
        Mathematics.douglas_peucker_r_density(points, splits, selected, end, threshold)

        splits[selected][1] = largest

    def douglas_peucker_density(points, splits, minimum_threshold):
        splits.clear()

        for ii in range(len(points)):
            splits.append([ii, 0])

        splits[0][1] = float('inf')
        splits[len(splits) - 1][1] = float('inf')
        
        Mathematics.douglas_peucker_r_density(points, splits, 0, len(points) - 1, minimum_threshold)
        splits.sort(key=lambda x: x[1], reverse=True)

    @staticmethod
    def douglas_peucker_density_trajectory(trajectory, minimum_threshold):
        splits = []
        indicies = []
        output = []

        splits.clear()
        output.clear()
        indicies.clear()

        Mathematics.douglas_peucker_density(trajectory, splits, minimum_threshold)

        ret = float('-inf')

        for split in splits:
            idx, score = split
            if score < minimum_threshold:
                continue
            indicies.append(idx)

        indicies.sort()

        for idx in indicies:
            output.append(trajectory[idx])

        return ret, output
    
    @staticmethod
    def vectorize(trajectory, normalize=True):
        vectors = []
        for ii in range(1, len(trajectory)):
            vec = trajectory[ii] - trajectory[ii - 1]
            if normalize:
                vec = vec.normalize()
            vectors.append(vec)
        return vectors
    
    @staticmethod
    def path_length(points):
        ret = 0
        for ii in range(1, len(points)):
            ret += points[ii].l2norm(points[ii - 1])
        return ret