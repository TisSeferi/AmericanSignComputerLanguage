from JkBlades import JkBlades
from Vector import Vector
import mathematics
import math


class JkFeatures:
    def __init__(self, blades=JkBlades, points=None, debug_print=False):
        self.pts = points
        self.vecs = Vector([])
        self.normalized_vecs = Vector([])
        m = points[0].size()
        
        # Store information about first frame to support static pose gestures. Calculate first frame position,
        # centroid of hand, and distance vectors for the first frame joints relative to centroid
        self.first_frame = self.pts[0]        
        self.ff_centroid = mathematics.calculate_centroid(self.first_frame)
        self.ff_joint_vecs_flat, self.ff_joint_vecs = mathematics.convert_joint_positions_to_distance_vectors(self.first_frame, self.ff_centroid)        

        self.pts = mathematics.resample(points=Vector(self.pts), n=blades.resample_cnt)
        self.path_length = mathematics.path_length(self.pts) 
        
        minimum = Vector(self.pts[0])
        maximum = Vector(self.pts[0])
        
        self.abs = Vector(0.0, m)

        for ii in range(1, blades.resample_cnt):
            vec = self.pts[ii] - self.pts[ii - 1]

            for jj in range(m):
                self.abs[jj] += abs(vec[jj])
                minimum[jj] = min(minimum[jj], self.pts[ii][jj])
                maximum[jj] = max(maximum[jj], self.pts[ii][jj])

            if blades.inner_product:
                vec = vec.normalize()
                self.vecs.append(vec)

            elif blades.euclidean_distance:
                if ii == 1:
                    self.vecs.append(Vector(self.pts[0]))

                self.vecs.append(Vector(self.pts[ii]))

            else:
                assert 0

        if blades.z_normalize:
            self.vecs = mathematics.z_normalize(self.vecs)
       
        if debug_print:
            print("\n=== Movement Debug Values ===")
            print("Before normalization:")
            print(f"Absolute movement raw values: {[f'{x:.4f}' for x in self.abs.data]}")
        
        bb_raw = (maximum - minimum)
        bb_magnitude = math.sqrt(sum(x*x for x in bb_raw.data))

        movement_ratio = self.path_length / bb_magnitude

        if movement_ratio > 2.0:
            print(f"This is a dynamic gesture (movement ratio: {movement_ratio:.2f})")
        else:
            print(f"This is a static gesture (movement ratio: {movement_ratio:.2f})")


        self.abs = self.abs.normalize()
        self.bb = bb_raw.normalize()


        if debug_print:
            print("\n")
            print(f"After abs normalization: {[f'{x:.4f}' for x in self.abs.data]}")        
           
            print("\n=== Bounding Box Analysis ===")
            print(f"Box dimensions (width, height, depth):")
            print(f"X (width): {bb_raw.data[0]:.4f} units")
            print(f"Y (height): {bb_raw.data[1]:.4f} units")
            print(f"Z (depth): {bb_raw.data[2]:.4f} units")
        
            # Calculate diagonal length for overall movement size
            print(f"\nTotal movement space (diagonal): {bb_magnitude:.4f} units")

            print("\nNormalized dimensions (relative proportions):")
            print(f"X: {self.bb.data[0]:.4f}")
            print(f"Y: {self.bb.data[1]:.4f}")
            print(f"Z: {self.bb.data[2]:.4f}")

            print()


