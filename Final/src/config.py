import os
from models import gradient_boosting

class Config(object):
    def __init__(self, model, feature):
        """setup configuration before executing run_model.py
        Attributes
        ----------
        data_dir:       string
                        raw data directory

        obj_dir:        string
                        object directory

        output_dir:     string
                        output directory

        feature_dir:    string
                        feature directory

        ans_dir:        string
                        y input directory

        id_dir:         string
                        id directory contains test_id and valid_id

        model:          object
                        specifies the regression model

        feature:        string
                        feature name
        """
        self.data_dir = os.path.join('/tmp2', 'b02902074', 'data')
        self.obj_dir = os.path.join('/tmp2', 'b02902015', 'ir', 'obj')
        self.output_dir = os.path.join('/tmp2', 'b02902015', 'ir', 'output')
        self.feature_dir = os.path.join(self.obj_dir, 'feature')
        self.ans_dir = os.path.join(self.obj_dir, 'ans')
        self.id_dir = os.path.join(self.obj_dir, 'id')
        self.model = model
        self.feature = feature

    def get_params(self):
        params = ', '.join([
            'model={}'.format(type(self.model).__name__),
            'feature={}'.format(self.feature),
            self.model.get_params()
            ])
        return params

config = Config(
        feature = 'label',
        # model = knn.KNeighbors(n_neighbors=50)
        model = gradient_boosting.GradientBoosting(max_depth=5, n_estimators=100)
        # model = adaboost.Adaboost(n_estimators=1000)
        # model = random_forest.RandomForest(n_estimators=1200)
        )
