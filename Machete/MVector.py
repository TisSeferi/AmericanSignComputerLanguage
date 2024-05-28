import math

class Vector:
    def __init__(self, data, m=None):
        if isinstance(data, list):
            self.data = data.copy()
        elif isinstance(data, int):
            self.data = [0.0] * data
        elif isinstance(data, (int, float)) and isinstance(m, int):
            self.data = [data] * m
        elif isinstance(data, Vector) and isinstance(m, Vector) and isinstance(m, float):
            if len(data.data) != len(m.data):
                raise ValueError("The size of the two vectors must be equal")
            self.data = [(1.0 - m) * data.data[i] + m * m.data[i] for i in range(len(data.data))]
        else:
            raise ValueError("Invalid arguments for Vector initialization")

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def set(self, rhs):
        for i in range(self.size()):
            self.data[i] = rhs

    def __neg__(self):
        return Vector([-x for x in self.data])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector([x * other for x in self.data])
        else:
            raise ValueError("Unsupported operand type")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector([x / other for x in self.data])
        else:
            raise ValueError("Unsupported operand type")

    def __add__(self, other):
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be the same size for addition")
            return Vector([self.data[i] + other.data[i] for i in range(len(self))])
        else:
            raise ValueError("Unsupported operand type")

    def __sub__(self, other):
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be the same size for subtraction")
            return Vector([self.data[i] - other.data[i] for i in range(len(self))])
        else:
            raise ValueError("Unsupported operand type")

    def __eq__(self, other):
        return isinstance(other, Vector) and self.data == other.data

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        result = self[0].__hash__()
        for i in range(1, self.size):
            result ^= self[i].__hash__()
        return result
 
    def l2norm2(self, other):
        ret = 0
        for i in range(0, self.size):
            delta = self.data[i] - other.data[i]
            ret += delta * delta
        return ret

    def l2norm(self, other=None):
        ret = 0.0
        for ii in range(0, self.size):
            ret += self.data[ii] * self.data[ii]
        return math.sqrt(ret)

    def length(self):
        ret = 0
        for i in range(0, self.size):
            ret += self.data[i] ** 2
        return math.sqrt(ret)


    def normalize(self):
        length = self.length()
        for i in range(0, self.size):
            self.data[i] /= length
        return self

    def dot(self, rhs):
        ret = 0
        for i in range(0, self.size):
            ret += self.data[i] * rhs[i]
        return ret

    def sum(self):
        ret = 0
        for i in range(0, self.size):
            ret += self.data[i]
        return ret

    def cumulative_sum(self):
        sum = 0
        for i in range(0, self.size):
            sum += self.data[i]
            self.data[i] = sum

    def clone(self):
        return Vector(self.data.copy())

    def append(self, value):
        self.data.append(value)

    def set_all_elements_to(self, value):
        for ii in range(len(self.data)):
            self.data[ii] = value

    def pop(self, index=-1):
        return self.data.pop(index)
    
    def clone(self):
        return Vector(self.data.copy())

    def is_zero(self):
        for ii in range(0, self.size):
            if self.data[ii] != 0.0:
                return False
        return True

    def minimum(self, other):
        for ii in range(0, self.size):
            if self.data[ii] < other.data[ii]:
                self.data = other.data

    def maximum(self, other):
        for ii in range(0, self.size):
            if self.data[ii] > other.data[ii]:
                self.data = other.data