import numpy as np
import random as r
import mathematics
import math

def flatten(negative):
    shape = np.shape(negative)
    dimensions = len(shape)
    
    if(dimensions < 3):
        print("Cannot flatten")
        return(negative)
    
    dim = 1
    for i in range(1, dimensions):
        dim *= shape[i]

    shape = (shape[0], dim)
    developed = np.zeros(shape)
    for index, frame in enumerate(negative):
        developed[index] = frame.flatten()

    #print(len(np.shape(developed)))
    return developed

def normalize(points):
    norm = np.linalg.norm(points)
    return points / norm

def z_normalize(points):
    mean = np.mean(points)
    std_val = np.std(points)

    z_scores = (points - mean) / std_val
    return(z_scores)

def path_length(points):
    total_len = 0.0
    for i in range(1, len(points)):
        total_len += np.linalg.norm(points[i] - points[i - 1])
    return total_len


def resample(points, n, variance=0):
    """
    Resamples a set of points using linear interpolation or nearest neighbor.

    Args:
        points (list of tuples or NumPy arrays): Original points.
        n (int): Number of desired resampled points.
        variance (float, optional): Variance for stochastic resampling (default is 0).

    Returns:
        list of tuples or NumPy arrays: Resampled points.
    """
    path_distance = path_length(points)
    intervals = np.zeros(n-1)

    # Uniform resampling
    if variance == 0.0:
        intervals.fill(1.0 / (n - 1))
    # Stochastic resampling
    else:
        b = np.sqrt(12 * variance)
        rr = np.random.rand(n - 1)
        intervals = 1.0 + rr * b
        intervals /= intervals.sum()

    assert np.isclose(intervals.sum(), 1.0, atol=1e-5)

    ret = [points[0]]
    prev = points[0]
    remaining_distance = path_distance * intervals[0]

    ii = 1
    #print("Mathematics resample")
    #print(len(points))
    while ii < len(points):
        distance = np.linalg.norm(points[ii] - prev)

        if distance < remaining_distance:
            prev = points[ii]
            remaining_distance -= distance
            ii += 1
            continue

        # Interpolate between the last point and the current point
        ratio = remaining_distance / distance
        ratio = np.clip(ratio, 0, 1)
        
        interpolated_point = prev + ratio * (points[ii] - prev)
        #print(np.shape(ret))

        ret.append(interpolated_point)

        if len(ret) == n:
            break

        prev = interpolated_point
        remaining_distance = path_distance * intervals[min(len(ret), len(intervals) - 1)]

    
    #print(len(ret))
    while len(ret) < n:
        ret.append(points[ii-1])

    return ret

##TODO Write GPSR


def gpsr(points, n, variance, remove_cnt) :
    resampled = resample(points, n + remove_cnt, variance)
    print(resampled)

    # Remove random points to simulate cutting corners.
    for ii in range(remove_cnt):
        remove_idx = r.randint(start = 0, stop = 65535)
        remove_idx = math.floor(remove_idx % math.floor(n + remove_cnt - ii))
        resampled.splice(remove_idx, 1)
    

    # Construct synthetic variation.
    m = resampled[0].data.length
    ret = np.zeros(m)

    for ii in range(1, resample.length):
        delta = resampled[ii] - resampled[ii - 1]
        ret[ii] = mathematics.normalize(delta)
    

# x = resample(points = np.load('templates/down-1.npy'), n = 6, variance=.1)
# print(x)
