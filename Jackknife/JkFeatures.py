import FeedData as fd
import numpy as np
from JkBlades import JkBlades
from Vector import Vector
import mathematics


class JkFeatures:
    def __init__(self, blades=JkBlades, points=None):
        self.pts = points
        self.vecs = Vector([])

        m = points[0].size()
        # print(m)
        
        self.pts = mathematics.resample(points=self.pts, n=blades.resample_cnt)
        # print(self.pts)

        minimum = Vector(self.pts[0])
        maximum = Vector(self.pts[0])

        self.abs = Vector(0.0, m)

        # print(np.shape(self.pts))

        for ii in range(1, blades.resample_cnt):
            vec = self.pts[ii].subtract(self.pts[ii - 1])

            for jj in range(m):
                self.abs[jj] += abs(vec[jj])
                minimum[jj] = min(minimum[jj], self.pts[ii][jj])
                maximum[jj] = max(maximum[jj], self.pts[ii][jj])

            if blades.inner_product:
                vec.normalize()
                self.vecs.append(vec)

            elif blades.euclidean_distance:
                if ii == 1:
                    self.vecs.append(Vector(self.pts[0]))

                self.vecs.append(Vector(self.pts[ii]))

            else:
                assert 0

        if blades.z_normalize:
            self.vecs = mathematics.z_normalize(self.vecs)

        self.abs.normalize()
        self.bb = maximum.subtract(minimum).normalize
