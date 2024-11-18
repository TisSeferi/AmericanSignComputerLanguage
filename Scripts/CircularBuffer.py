class CircularBuffer:
    def __init__(self, size=0):
        self.head = 0
        self.tail = 0
        self._size = size
        self.data = [None] * self._size

    def size(self):
        return self._size
    
    def count(self):
        ret = self.tail - self.head
        if ret < 0:
            ret += self._size

        return ret
    
    def resize(self, size):
        self.head = 0
        self.tail = 0
        self._size = size
        self.data = [None] * self._size

    def insert(self, item):
        self.data[self.tail] = item
        self.tail = (self.tail + 1) % self._size
        if self.tail == self.head:
            self.head = (self.head + 1) % self._size

    def pop(self):
        if self.head == self.tail:
            raise IndexError("Pop from empty buffer")
        result = self.data[self.head]
        self.head = (self.head + 1) % self._size
        return result

    def clear(self):
        self.head = 0
        self.tail = 0

    def empty(self):
        return self.head == self.tail

    def full(self):
        return self.head == ((self.tail + 1) % self._size)

    def get_value(self, idx):
        if idx < 0:
            idx = self.tail + idx
            if idx < 0:
                idx += self._size
        else:
            idx += self.head
        idx = idx % self._size
        return self.data[idx]

    def set_value(self, idx, value):
        if idx < 0:
            idx = self.tail + idx
            if idx < 0:
                idx += self._size
        else:
            idx += self.head
        idx = idx % self._size
        self.data[idx] = value

    def __getitem__(self, idx):
        return self.get_value(idx)

    def __setitem__(self, idx, value):
        self.set_value(idx, value)

    def copy(self, l_out, start, end):
        for idx in range(start, end + 1):
            l_out.append(self[idx])