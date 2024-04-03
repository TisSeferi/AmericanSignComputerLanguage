import numpy as np

class Vector:
    def __init__(self, x, size=None):
        if isinstance(x, Vector):
            self.data = x.data.copy()
        elif isinstance(x, (list, np.ndarray)):
            self.data = np.array(x, dtype=float)
        elif isinstance(x, (int, float)) and size is not None:
            self.data = np.full(size, x, dtype=float)
        elif isinstance(x, int) and size is None:
            self.data = np.zeros(x, dtype=float)
        else:
            self.data = x
            #raise ValueError("Invalid initialization parameters for Vector.")

    def negative(self):
        return Vector(-self.data)

    def add(self, other):
        return Vector(self.data + other.data)

    def subtract(self, other):
        return Vector(self.data - other.data)

    def divide(self, other):
        if isinstance(other, Vector):
            return Vector(self.data / other.data)
        else:
            return Vector(self.data / other)

    def multiply(self, other):
        if isinstance(other, Vector):
            return Vector(self.data * other.data)
        else:
            return Vector(self.data * other)

    def equals(self, other):
        return np.array_equal(self.data, other.data)

    def l2norm2(self, other):
        return np.sum((self.data - other.data) ** 2)

    def l2norm(self, other):
        return np.sqrt(self.l2norm2(other))

    def magnitude(self):
        return np.linalg.norm(self.data)

    def normalize(self):
        norm = self.magnitude()
        if norm > 0:
            self.data /= norm
        return self

    def dot(self, other):
        return np.dot(self.data, other.data)

    def sum(self):
        return np.sum(self.data)

    def cumulative_sum(self):
        self.data = np.cumsum(self.data)
        return self

    @staticmethod
    def interpolate_vectors(a, b, t):
        assert len(a.data) == len(b.data), "Vectors must be of the same dimension to interpolate."
        return Vector(a.data * (1 - t) + b.data * t)