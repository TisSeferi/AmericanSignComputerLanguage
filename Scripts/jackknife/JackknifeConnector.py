import numpy as np


class JKConnector:

    @staticmethod
    def get_jk_buffer_from_video(video, start_frame_no, end_frame_no):
        frames = []

        for ii in range(start_frame_no, end_frame_no + 1):
            try:
                v = video[ii]
            except IndexError:
                continue

            frames.append(np.asarray(v, dtype=np.float64))

        if frames:
            return np.array(frames)
        return np.empty((0, 63))
