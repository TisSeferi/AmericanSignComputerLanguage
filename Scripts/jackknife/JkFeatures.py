import numpy as np
from .JkBlades import JkBlades
import mathematics


class JkFeatures:
    def __init__(self, blades=JkBlades, points=None, is_template=False):
        # points: (n_frames, d) ndarray

        # First-frame features for static pose detection
        self.first_frame = points[0]  # (d,)
        self.ff_centroid = mathematics.calculate_centroid(self.first_frame)
        self.ff_joint_vecs_flat, self.ff_joint_vecs = (
            mathematics.convert_joint_positions_to_distance_vectors(
                self.first_frame, self.ff_centroid
            )
        )
        self.ff_bb_magnitude = mathematics.calculate_spatial_bb(self.first_frame)

        # Resample trajectory
        pts = mathematics.resample(points, n=blades.resample_cnt)  # (resample_cnt, d)
        self.path_length = mathematics.path_length(pts)

        # Absolute movement per dimension (sum of |delta| across all steps)
        diffs = np.diff(pts, axis=0)  # (resample_cnt-1, d)
        self.abs = np.sum(np.abs(diffs), axis=0)  # (d,)

        # Bounding box of resampled trajectory
        min_pt, max_pt = mathematics.bounding_box(pts)
        bb_raw = max_pt - min_pt  # (d,)

        # Build direction vectors (vecs)
        if blades.inner_product:
            norms = np.linalg.norm(diffs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            self.vecs = diffs / norms  # (resample_cnt-1, d)
        elif blades.euclidean_distance:
            self.vecs = pts  # (resample_cnt, d)
        else:
            assert False, "Either inner_product or euclidean_distance must be set"

        if blades.z_normalize:
            self.vecs = mathematics.z_normalize(self.vecs)

        if is_template:
            movement_ratio = self.path_length / self.ff_bb_magnitude
            self.is_static = movement_ratio <= 1.3

        # Normalize abs and bb for correction factors
        abs_norm = np.linalg.norm(self.abs)
        self.abs = self.abs / abs_norm if abs_norm > 1e-8 else self.abs

        bb_norm = np.linalg.norm(bb_raw)
        self.bb = bb_raw / bb_norm if bb_norm > 1e-8 else bb_raw
