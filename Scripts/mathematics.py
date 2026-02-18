import random as r
import numpy as np
import math


def flatten(arr):
    """Convert (n, d) ndarray to list of 1D row arrays."""
    return list(arr)


def z_normalize(pts):
    """Z-normalize a 2D ndarray column-wise (axis=0)."""
    std = pts.std(0, ddof=1)
    std = np.where(std < 1e-8, 1.0, std)
    return (pts - pts.mean(0)) / std


def path_length(pts):
    """Total Euclidean path length of a (n, d) ndarray."""
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def resample(pts, n=8, variance=None):
    """Resample pts (n_frames, d) ndarray to exactly n evenly-spaced points."""
    path_dist = path_length(pts)

    if not variance:
        intervals = np.ones(n - 1) / (n - 1)
    else:
        b = (12 * variance) ** 0.5
        intervals = 1.0 + np.array([r.random() for _ in range(n - 1)]) * b
        intervals = intervals / intervals.sum()

    assert abs(intervals.sum() - 1.0) < 1e-4

    remaining_distance = path_dist * intervals[0]
    prev = pts[0].copy()
    result = [pts[0].copy()]
    ii = 1

    while ii < len(pts):
        distance = float(np.linalg.norm(pts[ii] - prev))

        if distance < remaining_distance:
            prev = pts[ii]
            remaining_distance -= distance
            ii += 1
            continue

        ratio = remaining_distance / distance if distance > 0 else 1.0
        if ratio > 1.0 or math.isnan(ratio):
            ratio = 1.0

        new_pt = prev + ratio * (pts[ii] - prev)
        result.append(new_pt)

        if len(result) == n:
            return np.array(result)

        prev = result[-1]
        remaining_distance = path_dist * intervals[len(result) - 1]

    while len(result) < n:
        result.append(pts[ii - 1].copy())

    assert len(result) == n
    return np.array(result)


def gpsr(pts, n, variance, remove_cnt):
    """Generate a synthetic trajectory via GPSR.

    Returns (n, d) ndarray of direction vectors: a zero row followed by
    n-1 normalized consecutive differences.
    """
    resampled = resample(pts, n + remove_cnt, variance)

    idx_list = list(range(len(resampled)))
    for ii in range(remove_cnt):
        remove_idx = int(r.random() * 65535) % (n + remove_cnt - ii)
        idx_list.pop(remove_idx)
    resampled = resampled[idx_list]  # (n, d)

    d = resampled.shape[1]
    deltas = np.diff(resampled, axis=0)  # (n-1, d)
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normalized = deltas / norms  # (n-1, d)
    return np.vstack([np.zeros((1, d)), normalized])  # (n, d)


def bounding_box(pts):
    """Returns (min_point, max_point) as 1D ndarrays over axis=0."""
    return pts.min(0), pts.max(0)


def vectorize(pts, normalize=True):
    """Compute direction vectors from consecutive point differences.

    pts: (n, d) ndarray
    Returns: (n-1, d) ndarray (normalized rows if normalize=True)
    """
    vecs = np.diff(pts, axis=0)  # (n-1, d)
    if normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        vecs = vecs / norms
    return vecs


def calculate_centroid(frame):
    """Mean 3D position from flattened (63,) frame array. Returns (3,) ndarray."""
    return frame.reshape(-1, 3).mean(0)


def calculate_spatial_bb(frame):
    """Diagonal length of 3D bounding box from flattened (63,) frame array."""
    pts = frame.reshape(-1, 3)
    return float(np.linalg.norm(pts.max(0) - pts.min(0)))


def convert_joint_positions_to_distance_vectors(joints_xyz, centroid):
    """Convert flattened (63,) joint positions to normalized direction vectors from centroid.

    Returns: (flat_ndarray (63,), 2D_ndarray (21, 3))
    """
    pts = joints_xyz.reshape(-1, 3) - centroid  # (21, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normalized = pts / norms  # (21, 3)
    return normalized.flatten(), normalized  # (63,), (21, 3)


def calculate_joint_angle_disparity(a, b):
    """Per-joint dot products and total similarity between two flat (63,) direction arrays.

    Returns: (list of 21 per-joint dot products, total sum as float)
    """
    dots = np.sum(a.reshape(-1, 3) * b.reshape(-1, 3), axis=1)  # (21,)
    return dots.tolist(), float(dots.sum())


def douglas_peucker_r_density(points, splits, start, end, threshold):
    """Recursive density-based Douglas-Peucker. points is (n, d) ndarray."""
    if start + 1 > end:
        return

    AB = points[end] - points[start]
    denom = float(np.dot(AB, AB))

    largest = float('-inf')
    selected = -1

    for ii in range(start + 1, end):
        AC = points[ii] - points[start]
        numer = float(np.dot(AC, AB))
        if denom == 0.0:
            d2 = float(np.linalg.norm(AC))
        else:
            d2 = float(np.dot(AC, AC)) - numer * numer / denom

        v1 = points[ii] - points[start]
        v2 = points[end] - points[ii]

        l1 = float(np.linalg.norm(v1))
        l2 = float(np.linalg.norm(v2))

        denom_angle = l1 * l2 if l1 * l2 > 0 else 1.0
        dot = float(np.dot(v1, v2)) / denom_angle
        dot = max(-1.0, min(1.0, dot))
        angle = math.acos(dot)
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
    splits[-1][1] = float('inf')

    douglas_peucker_r_density(points, splits, 0, len(points) - 1, minimum_threshold)
    splits.sort(key=lambda x: x[1], reverse=True)


def douglas_peucker_density_trajectory(trajectory, minimum_threshold):
    """Run DP density simplification on trajectory (n, d) ndarray.

    Returns (ret, output) where output is a list of 1D ndarray rows.
    """
    splits = []
    indices = []
    output = []

    douglas_peucker_density(trajectory, splits, minimum_threshold)

    ret = float('-inf')

    for split in splits:
        idx, score = split
        if score < minimum_threshold:
            continue
        indices.append(idx)

    indices.sort()

    for idx in indices:
        output.append(trajectory[idx])

    return ret, output
