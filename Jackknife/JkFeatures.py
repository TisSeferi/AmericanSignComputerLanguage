import FeedData as fd
import numpy as np
from JkBlades import JkBlades
from Vector import Vector
import mathematics

class JkFeatures:
    def __init__(self, blades=JkBlades, points=None):
        self.pts = mathematics.flatten(points)
        self.vecs = []

        m = len(points[0].data)
        #print(m)

        self.pts = mathematics.resample(points=self.pts, n=blades.resample_cnt)

        minimum = Vector(self.pts[0].data)
        maximum = Vector(self.pts[0].data)

        self.abs = Vector(0.0, m)

        #print(np.shape(self.pts))

        for ii in range(1, blades.resample_cnt):
            vec = self.pts[ii].subtract(self.pts[ii - 1])

            for jj in range(m):
                self.abs.data[jj] += abs(vec.data[jj])
                minimum.data[jj] = min(minimum.data[jj], self.pts[ii].data[jj])
                maximum.data[jj] = max(maximum.data[jj], self.pts[ii].data[jj])

        if (blades.inner_product):
            self.vecs.append(Vector(vec.normalize()))

        elif (blades.euclidean_distance):
            if (ii == 1):
                self.vecs.append(Vector(self.pts[0]))

            self.vecs.append(Vector(self.pts[ii]))

        else:
            assert(0)

        if (blades.z_normalize):
           self.vecs = mathematics.z_normalize(self.vecs)

        self.abs = mathematics.normalize(self.abs)
        self.bb = (maximum.subtract(minimum).normalize)