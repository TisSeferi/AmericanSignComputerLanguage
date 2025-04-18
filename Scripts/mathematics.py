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

    mean = mean / (n)

    for ii in range(n):
        for jj in range(m):
            diff = points[ii][jj] - mean[jj]
            variance[jj] += diff ** 2

    variance = variance / (n-1)

    for ii in range(m):
        variance[ii] = variance[ii] ** .5

    for ii in range(n):
        points[ii] = (points[ii]-mean) / (variance)

    return points


def path_length(points):
    ret = 0.0
    pointssize = points.size() if isinstance(points, Vector) else len(points)
    for ii in range(1, pointssize):
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

        intervals = intervals / (intervals.sum())

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

    while ret.size() < n:
        ret.append(points[ii - 1])
    #print ("Resampled Length assertion Check:" +  str(ret.size()) + " " + str(n) + " " + str(points.size()) + " " + str(ii)) 

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
        min_point.minimum(trajectory[ii])
        max_point.maximum(trajectory[ii])

    return min_point, max_point

def douglas_peucker_r_density(points, splits, start, end, threshold):
    
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
            d2 = AC.magnitude()

        v1 = points[ii] - points[start]
        v2 = points[end] - points[ii]

        l1 = v1.magnitude()
        l2 = v2.magnitude()

        dot = v1.dot(v2)
        dot /= (l1 * l2 > 0) if (l1 * l2) else 1.0
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

def calculate_centroid(trajectory):
    # Create a new Vector filled with zeros to store the average position
    centroid = Vector([0.0, 0.0, 0.0])
    
    # Calculate how many points we have (each point has x,y,z so divide by 3)
    num_points = trajectory.size() // 3 
    
    for i in range(num_points):
        # Get the x,y,z coordinates for each point
        x, y, z = trajectory[i * 3], trajectory[i * 3 + 1], trajectory[i * 3 + 2]

        # Add each point's position to our running total
        centroid += Vector([x, y, z])
    
    # Divide by 21 to get the average position (center point)
    centroid = centroid / num_points
    
    return centroid

# Input frame is a single 21x3 array of joint positions flattened
def calculate_spatial_bb(frame):
    # Create a new Vector filled with zeros to store the average position
    minimum = Vector([float('inf'), float('inf'), float('inf')])
    maximum = Vector([float('-inf'), float('-inf'), float('-inf')])

    # Calculate how many points we have (each point has x,y,z so divide by 3)
    num_points = frame.size() // 3 
    
    for i in range(num_points):
        # Get the x,y,z coordinates for each point
        temp_vec = Vector([frame[i * 3], frame[i * 3 + 1], frame[i * 3 + 2]])

        # Update the minimum and maximum values for each axis    
        minimum.minimum(temp_vec)
        maximum.maximum(temp_vec)

    return (maximum - minimum).magnitude()

# Converts from a list of joint coordinates to a list of distance vectors from the centroid
def convert_joint_positions_to_distance_vectors(joints_xyz, centroid):
    # Create empty lists to store our results
    direction_vector_joints = Vector([])       # Will store 3D vectors
    flat_direction_vector_joints = Vector([])  # Will store flattened coordinates
    
    # Calculate number of points (each point has x,y,z coordinates)
    num_points = joints_xyz.size() // 3

    for i in range(num_points):

        x, y, z = joints_xyz[i * 3], joints_xyz[i * 3 + 1], joints_xyz[i * 3 + 2]        
        # Calculate the position of the joint relative to the centroid
        joint_position = Vector([x, y, z]) - centroid

        # Make the vector length 1.0 (normalize it)
        joint_position = joint_position.normalize()
        
        # Store the 3D vector
        direction_vector_joints.append(joint_position)
        
        # Append the distance vector to the list as we are dealing with flat vectors. May not be used.
        flat_direction_vector_joints.append(joint_position[0])
        flat_direction_vector_joints.append(joint_position[1])
        flat_direction_vector_joints.append(joint_position[2])


    return flat_direction_vector_joints, direction_vector_joints


# Calculate per joint angle between two lists(vectors) of direction vectors, and the summed angle for the entire pose
# Works with FLAT vectors of normalized direction vectors with three coords per joint.
def calculate_joint_angle_disparity(joint_vecs_a, joint_vecs_b):
    # Create lists to store our results
    joint_angles = []    # Will store angles between each pair of joints
    total_angle = 0.0   # Will store the sum of all angles
    
    # Make sure we have the same number of points in both sets
    assert joint_vecs_a.size() == joint_vecs_b.size(), "Input vectors must have the same length"

    # Calculate how many points we have
    num_points = joint_vecs_a.size() // 3

    for i in range(num_points):
        # Get the vectors for matching joints from both sets
        vec_a = Vector([joint_vecs_a[i * 3], joint_vecs_a[i * 3 + 1], joint_vecs_a[i * 3 + 2]])
        vec_b = Vector([joint_vecs_b[i * 3], joint_vecs_b[i * 3 + 1], joint_vecs_b[i * 3 + 2]])

        # Make sure vectors are normalized (length = 1)
        assert abs(vec_a.magnitude() - 1.0) < 0.01, "Input vector a must be normalized" 
        assert abs(vec_b.magnitude() - 1.0) < 0.01, "Input vector b must be normalized"

        # Calculate how similar the vectors are using dot product
        # 1.0 means identical direction, -1.0 means opposite direction
        dot_product = vec_a.dot(vec_b)                

        # Store the similarity score for this joint
        joint_angles.append(dot_product)
        
        # Add to our running total
        total_angle = total_angle + dot_product

    return joint_angles, total_angle