from sklearn import linear_model

class SGD(linear_model.SGDRegressor):
    def __init__(self, n_iter):
        super().__init__(n_iter = n_iter)

    def get_params(self):
        params = super().get_params()
        return ', '.join(['{}={}'.format(key, val) for key, val in params.items()])
