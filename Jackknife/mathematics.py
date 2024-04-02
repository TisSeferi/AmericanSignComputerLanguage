import numpy as np
import scipy

def z_normalize(points):
    mean = np.mean(points)
    std_val = np.std(points)

    z_scores = (points - mean) / std_val
    return(z_scores)

def path_length(points):
    return(np.linalg.norm(points))




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
    path_distance = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    intervals = np.zeros(n - 1)

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
    while ii < len(points):
        distance = np.linalg.norm(points[ii] - prev)

        if distance < remaining_distance:
            prev = points[ii]
            remaining_distance -= distance
            ii += 1
            continue

        # Interpolate between the last point and the current point
        ratio = remaining_distance / distance
        interpolated_point = prev + ratio * (points[ii] - prev)
        ret.append(interpolated_point)

        if len(ret) == n:
            break

        prev = interpolated_point
    
    return ret


x = resample(points = np.load('templates/down-1.npy'), n = 6, variance=.1)
print(x)
