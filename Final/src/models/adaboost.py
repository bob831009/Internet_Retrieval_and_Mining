from sklearn import ensemble

class AdaBoost(ensemble.AdaBoostRegressor):
    def __init__(self, n_estimators):
        super().__init__(n_estimators = n_estimators)

    def get_params(self):
        params = super().get_params()
        return ', '.join(['{}={}'.format(key, val) for key, val in params.items()])
