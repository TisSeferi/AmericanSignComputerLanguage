from JkBlades import JkBlades
from Vector import Vector
import mathematics
import math


class JkFeatures:
    def __init__(self, blades=JkBlades, points=None, debug_print=False):
        self.pts = points
        self.vecs = Vector([])

        m = points[0].size()
        
        
        self.pts = mathematics.resample(points=Vector(self.pts), n=blades.resample_cnt)
        # print(self.pts)

        minimum = Vector(self.pts[0])
        maximum = Vector(self.pts[0])

        self.abs = Vector(0.0, m)

        # print(np.shape(self.pts))

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
        
        self.abs = self.abs.normalize()
        bb_raw = (maximum - minimum)
        diagonal = math.sqrt(sum(x*x for x in bb_raw.data))
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
            print(f"\nTotal movement space (diagonal): {diagonal:.4f} units")

            print("\nNormalized dimensions (relative proportions):")
            print(f"X: {self.bb.data[0]:.4f}")
            print(f"Y: {self.bb.data[1]:.4f}")
            print(f"Z: {self.bb.data[2]:.4f}")

            print()

        # Determine if the gesture is static or continuous
        threshold = 0.06  # Define your threshold
        is_static = self.is_static_gesture(diagonal, threshold)

        if is_static:
            print("The gesture is static.")
        else:
            print("The gesture is continuous.")

    def is_static_gesture(self, diagonal, threshold):
        """Determine if the gesture is static or continuous based on the diagonal length."""
        return diagonal < threshold