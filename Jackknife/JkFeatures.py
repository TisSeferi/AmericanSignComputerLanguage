import FeedData as fd
import numpy as np
from JkBlades import JkBlades
from Vector import Vector
import mathematics

class JkFeatures:
    def __init__(self, blades=JkBlades, points=None):
        self.pts = mathematics.flatten(points)
        self.vecs = []

        m = np.shape(self.pts)[1]
        #print(m)
        self.pts = mathematics.resample(points=self.pts, n=blades.resample_cnt)

        minimum = self.pts[0]
        maximum = self.pts[0]

        self.abs = np.zeros((m))

        print(np.shape(self.pts))

        for ii in range(1, blades.resample_cnt):
            vec = self.pts[ii] - self.pts[ii - 1]

            minimum = np.minimum(minimum, self.pts[ii])

            for jj in range(m):
                self.abs[jj] = self.abs[jj] + np.abs(vec[jj])
            
            if (blades.inner_product):
                self.vecs.append(mathematics.normalize(vec))
            elif (blades.euclidean_distance):
                if (ii == 1):
                    self.vecs.append(self.pts[0])

                    self.vecs.append(self.pts[ii])
                else:
                    assert(0)

        self.vecs = np.array(self.vecs)

        if (blades.z_normalize):
           self.vecs = mathematics.z_normalize(self.vecs)

        self.abs = mathematics.normalize(self.abs)
        self.bb = mathematics.normalize(maximum - minimum)