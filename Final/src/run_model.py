import os
import numpy as np
from config import config
from datetime import datetime
import timeit
import pickle

def output_result(y_pred, model, cur_time):
    """saves model and generates result.csv

    parameters
    ----------
    y_pred:     array-like of shape = [n_samples]
                The predicted result

    model:      object
                The regression model

    cur_time:   string
                current time
    """

    # create ouput_dir if directory does not exist
    output_dir = config.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # create directory with the current date name
    result_dir = os.path.join(output_dir, cur_time)
    os.mkdir(result_dir)

    # save model
    pickle.dump(model, open(os.path.join(result_dir, 'model.pkl'), 'wb'))

    # generates result.csv
    test_ids = np.load(os.path.join(config.id_dir, 'test_ids.npy'))
    with open(os.path.join(result_dir, 'result.csv'), 'w') as fp:
        for id, y in zip(test_ids, y_pred):
            fp.write('{},{}\n'.format(id, y))

def write_log(cur_time, valid_score, test_score, cost_time):
    with open('log', 'a') as fp:
        fp.write( '{}\t{}\tvalid_score={}\ttest_score={}\t{:.2f}min.\n'.format(
            cur_time, config.config_msg(), valid_score, test_score, cost_time))

def main():
    """run_model.py loads the params from config.py """

    feature_dir = config.feature_dir
    ans = config.ans_dir

    print('loading data...')
    X_train = np.load(os.path.join(feature_dir,'X_train.npy'))
    X_valid = np.load(os.path.join(feature_dir,'X_valid.npy'))
    X_test = np.load(os.path.join(feature_dir,'X_test.npy'))
    y_train = np.load(os.path.join(ans_dir,'y_train.npy'))
    y_valid = np.load(os.path.join(ans_dir,'y_valid.npy'))
    y_test = np.load(os.path.join(ans_dir,'y_test.npy'))
    
    # load model from config
    clf = config.model
    start_time = timeit.default_timer()
    print('start training {}...'.format(config.get_params()))
    clf.train(X_train, y_train)
    end_time = timeit.default_timer()
    cost_time = (end_time - start_time) / 60.
    print('{} ran for {:.2f}min'.format(type(clf).__name__, cost_time))

    valid_score = clf.score(X_valid, y_valid)
    print('validation score:', valid_score)

    print('start predicting test data...')
    y_pred = clf.predict(X_test)
    test_score = clf.score(X_test, y_test)
    print('validation score:', valid_score)

    cur_time = datetime.now().strftime('%m-%d_%H:%M')
    output_result(y_pred, clf, cur_time)
    write_log(cur_time, valid_score, test_score, cost_time)

if __name__ == '__main__':
    main()
