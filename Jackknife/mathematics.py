import random as r
from Vector import Vector 
import numpy as np
import math



def flatten(negative):
    shape = np.shape(negative)
    dimensions = len(shape)

    if dimensions < 3:
        print("Cannot flatten")
        return (negative)

    dim = 1
    for i in range(1, dimensions):
        dim *= shape[i]

    shape = (shape[0], dim)
    developed = []
    for index, frame in enumerate(negative):
        developed.append(Vector(frame.flatten().tolist()))
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


