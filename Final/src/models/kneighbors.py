from sklearn import neighbors

class KNeighbors(neighbors.KNeighborsRegressor):
    def __init__(self, n_neighbors, n_jobs=-1):
        super().__init__(n_neighbors = n_neighbors, n_jobs = n_jobs)

    def get_params(self):
        params = super().get_params()
        return ', '.join(['{}={}'.format(key, val) for key, val in params.items()])
