import numpy as np

class Vector:
    def __init__(self, x, size=None):
        if isinstance(x, Vector):
            self.data = x.data.copy()
        elif isinstance(x, list):
            print(x)
            print("Thanks for finding the exception")
        elif isinstance(x, np.ndarray):
            self.data = np.array(x, dtype=float)
        elif isinstance(x, (int, float)) and size is not None:
            self.data = np.full(size, x, dtype=float)
        elif isinstance(x, int) and size is None:
            self.data = np.zeros(x, dtype=float)
        else:
            self.data = x
            #raise ValueError("Invalid initialization parameters for Vector.")

    def length(self):
        return len(self.data)
    
    def size(self):
        return len(self.data)
    
    def elementAt(self, idx):
        return self.data[idx]
    
    def setAllElementsTo(self, rhs):
        for ii in range(self.length):
            self.data[ii] = rhs

    def negative(self):
        m = self.length()

        vec = Vector(m)
        for ii in range(m):
            vec.data[ii] = -self.data[ii]

        return vec
    
    def add(self, rhs):
        return self.data + rhs.data

    def subtract(self, rhs):
        return self.data - rhs.data
    
    def divide(self, rhs):
        if isinstance(rhs, Vector):
            rhs = rhs.data
        
        return self.data / rhs
    
    def multiply(self, rhs):
        if isinstance(rhs, Vector):
            rhs = rhs.data
        
        return self.data * rhs
    
    def equals(self, rhs):
        m = self.length()
        ret = True

        for ii in range(m):
            ret = ret.data[ii] and rhs.data[ii]

        return ret
    
    def l2norm2(self, other):
        m = self.length()
        ret = 0

        for ii in range(m):
            ret += (self.data[ii] - other.data[ii]) ** 2

        return ret

    def l2norm(self, other):
        return self.l2norm2(other) ** .5
    
    def magnitude(self):
        m = self.length()
        ret = 0
        for ii in range(m):
            ret += self.data[ii] ** 2

        return ret ** .5
    
    def normalize(self):
        len = self.magnitude()

        self.data = self.data / len

    def dot(self, rhs):
        m = self.length()
        ret = 0

        for ii in range():
            ret += self.data[ii] * rhs.data[ii]

        return ret
    
    def sum(self):
        ret = 0
        for ii in self.length():
            ret += self.data[ii]
        return ret
    
    def cumulative_sum(self):
        ret = 0

        for ii in range(self.length()):
            ret += self.data[ii]
            self.data[ii] = ret

    def InterpolateVectors(a, b, t):
        m = a.length()
        n = b.length()

        #console.assert(m == n, 'Different sized arrays to interpolate')

        data = np.zeros(m)
        for ii in range (0, m):
            data[ii] = (1.0 - t) * a.data[ii]
            data[ii] += t * b.data[ii]

        return Vector(data)
    
