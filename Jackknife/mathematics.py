import random as r
from Vector import Vector 
import numpy as np
import math

def flatten(negative):
    developed = []
    for index, frame in enumerate(negative):
        developed.append(Vector(frame.tolist()))
    return developed


def z_normalize(points):
    # print("math 29")
    # print(points)
    n = points.size()
    m = points[0].size()

    mean = Vector(0.0, m)
    variance = Vector(0.0, m)

    for ii in range(n):
        mean += points[ii]

    mean = mean.__div__(n)

    for ii in range(n):
        for jj in range(m):
            diff = points[ii][jj] - mean[jj]
            variance[jj] += diff ** 2

    variance = variance.__div__(n-1)

    for ii in range(m):
        variance[ii] = variance[ii] ** .5

    for ii in range(n):
        points[ii] = (points[ii]-mean).__div__(variance)

    return points


def path_length(points):
    ret = 0.0
    for ii in range(1, points.size()):
        ret += points[ii].l2norm(points[ii - 1])

    return ret


def resample(points, n=8, variance=None):
    path_distance = path_length(points)
    intervals = Vector(n - 1)

    interval = None
    ii = None

    if not variance:
        intervals.set_all_elements_to(1.0 / (n - 1))
    else:
        for ii in range(n - 1):
            b = (12 * variance) ** .5
            rr = r.random()
            intervals.data[ii] = 1.0 + rr * b

        intervals = intervals.__div__(intervals.sum())

    assert abs(intervals.sum() - 1 < .00001)

    remaining_distance = path_distance * intervals[0]
    prev = points[0]
    ret = Vector([Vector(points[0])])
    ii = 1

    while ii < points.size():        
        distance = points[ii].l2norm(prev)

        if distance < remaining_distance:
            prev = points[ii]
            remaining_distance -= distance
            ii += 1
            continue
        ratio = remaining_distance / distance

        if ratio > 1.0 or math.isnan(ratio):
            ratio = 1.0

        ret.append(
            Vector.interpolate(
                prev, points[ii], ratio
            )
        )


        if ret.size() == n:
            return ret

        prev = ret[ret.size() - 1]

        remaining_distance = path_distance * intervals[(ret.size() - 1)]

    if ret.size() < n:
        ret.append(points[ii - 1])

    assert ret.size() == n
    return ret

def gpsr(points, n, variance, remove_cnt):
    ret = Vector([])
    resampled = resample(points, n + remove_cnt, variance)

    for ii in range(remove_cnt):
        remove_idx = r.random() * 65535
        remove_idx = math.floor(remove_idx % math.floor(n + remove_cnt - ii))

        resampled.pop(remove_idx)

    m = resampled[0].size()
    ret.append(Vector(0.0, m))

    for ii in range(resampled.size()):
        delta = resampled[ii] - resampled[ii - 1]
        ret.append(delta.normalize())

    return ret

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

    douglas_peucker_r_density(points, splits, start, selected, threshold)
    douglas_peucker_r_density(points, splits, selected, end, threshold)

    splits[selected][1] = largest

def douglas_peucker_density(points, splits, minimum_threshold):
    splits.clear()

    for ii in range(len(points)):
        splits.append([ii, 0])

    splits[0][1] = float('inf')
    splits[len(splits) - 1][1] = float('inf')
    
    douglas_peucker_r_density(points, splits, 0, len(points) - 1, minimum_threshold)
    splits.sort(key=lambda x: x[1], reverse=True)

@staticmethod
def douglas_peucker_density_trajectory(trajectory, minimum_threshold):
    splits = []
    indicies = []
    output = []

    splits.clear()
    output.clear()
    indicies.clear()

    douglas_peucker_density(trajectory, splits, minimum_threshold)

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

