
class Mathematics:
    def __init__(self) -> None:
        pass

    @staticmethod
    def vectorize(trajectory, normalize=True):
        vectors = []
        for ii in range(1, len(trajectory)):
            vec = trajectory[ii] - trajectory[ii - 1]
            if normalize:
                vec = vec.normalize()
            vectors.append(vec)
        return vectors