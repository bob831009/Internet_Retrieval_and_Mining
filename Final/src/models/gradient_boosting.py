from sklearn import ensemble

class GradientBoosting(ensemble.GradientBoostingRegressor):
    def __init__(self, n_estimators, max_depth):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth)
        
    def get_params(self):
        params = super().get_params()
        return ', '.join(['{}={}'.format(key, val) for key, val in params.items()])
