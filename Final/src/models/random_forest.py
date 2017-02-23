from sklearn import ensemble

class RandomForest(ensemble.RandomForestRegressor):
    def __init__(self, n_estimators, max_depth, oob_score=False, n_jobs=-1):
        super().__init__(
                n_estimators = n_estimators, 
                max_depth = max_depth, 
                oob_score = oob_score, 
                n_jobs = n_jobs
                )

    def get_params(self):
        params = super().get_params()
        return ', '.join(['{}={}'.format(key, val) for key, val in params.items()])
