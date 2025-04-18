from Vector import Vector

class JKConnector:

    def get_jk_buffer_from_video(video, start_frame_no, end_frame_no):
        buffer = []

        for ii in range(start_frame_no, end_frame_no + 1):
            try:
                v = video[ii]
            except IndexError:
                # Skip over frames that are out of bounds.
                continue

            JKVectorData = []

            for data in v:
                JKVectorData.append(data)

            jv = Vector(JKVectorData)
            buffer.append(jv)

        return buffer            