import numpy as np

class Vector:
    def __init__(self, x, size=None):
        if isinstance(x, int) and size is None:
            self.data = np.zeros(x)
        elif isinstance(x, Vector):
            self.data = x.data.copy()
        elif isinstance(x, np.ndarray):
            self.data = x.copy()
        elif isinstance(x, list) or isinstance(x, float) or isinstance(x, int):
            self.data = np.array(x)
        else:
            raise ValueError("Invalid Vector initialization parameters.")
        
        self.length = len(self.data)

    def subtract(self, other):
        return Vector(self.data - other.data)

    def add(self, other):
        return Vector(self.data + other.data)

    def dot(self, other):
        return np.dot(self.data, other.data)

    def l2norm2(self, other):
        return np.sum((self.data - other.data) ** 2)

    def normalize(self):
        mag = np.linalg.norm(self.data)
        if mag > 0:
            self.data /= mag
        return self  