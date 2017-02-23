import os
import json
import pickle
import timeit
import datetime
import numpy as np

from config import config

def extract_semantic_feature(train_data, valid_data, test_data, config):
    Vocab_cut_dir = os.path.join('/tmp2', 'b02902015', 'vocab_cut.json');
    Vocab_emotion_dir = os.path.join('/tmp2', 'b02902015', 'vocab_emotion.json');
    
    Vocab_cut = json.load(open(Vocab_cut_dir));
    Vocab_emotion = json.load(open(Vocab_emotion_dir));

    train_emotion = [];
    valid_emotion = [];
    test_emotion = [];

    Message = ['train_data', 'valid_data', 'test_data'];
    Total_data = [train_data, valid_data, test_data];
    Total_feature = [[], [], []];

    for i in range(3):
        print("Handling %s Emotion!" % (Message[i]));
        data  = Total_data[i];
        for Doc_obj in data:
            Tag_Score = 0;
            Tag_Term_Num = 0;
            for term in Doc_obj['tag']:
                if(term in Vocab_emotion):
                    Tag_Score += Vocab_emotion[term];
                    Tag_Term_Num += 1;
            if(Tag_Term_Num == 0):
                Tag_Score = 0;
            else:
                Tag_Score /= Tag_Term_Num;

            Title_Score = 0;
            Title_term_num = 0;
            Title = Doc_obj['Title'].split(" ");
            for term in Title:
                if(term in Vocab_emotion):
                    Title_Score += Vocab_emotion[term];
                    Title_term_num += 1;
            if(Title_term_num == 0):
                Title_Score = 0;
            else:
                Title_Score /= Title_term_num;

            MainContent_Score = 0;
            MainContent_term_num = 0;
            MainContent = Doc_obj['main_content'].split(" ");
            for term in MainContent:
                if(term in Vocab_emotion):
                    MainContent_Score += Vocab_emotion[term];
                    MainContent_term_num += 1;
            if(MainContent_term_num == 0):
                MainContent_Score = 0;
            else:
                MainContent_Score /= MainContent_term_num;

            Total_feature[i].append([Title_Score, Tag_Score, MainContent_Score]);

    np.save(os.path.join(config.feature_dir, 'train_semantic'), Total_feature[0]);
    np.save(os.path.join(config.feature_dir, 'valid_semantic'), Total_feature[1]);
    np.save(os.path.join(config.feature_dir, 'test_semantic'), Total_feature[2]);



def extract_base_feature(train_data, valid_data, test_data, config):
    pass

def extract_ids_ans(train_data, valid_data, test_data, config):
    """extract document ids """

    print('start extracting ids and answers...')
    if not os.path.exists(config.id_dir):
        os.mkdir(config.id_dir)
    if not os.path.exists(config.ans_dir):
        os.mkdir(config.ans_dir)

    ids = []
    ans = []
    for data in (train_data + valid_data + test_data):
        ids.append(data['ID'])
        ans.append(data['messageNum']['g'] - data['messageNum']['b'])
    
    # dump ids
    np.save(
            os.path.join(config.id_dir, 'train_ids'), 
            ids[:len(train_data)]
            )
    np.save(
            os.path.join(config.id_dir, 'valid_ids'),
            ids[len(train_data):len(train_data) + len(valid_data)]
            )
    np.save(
            os.path.join(config.id_dir, 'test_ids'),
            ids[len(train_data) + len(valid_data):]
            )

    # dump ans
    np.save(
            os.path.join(config.ans_dir, 'y_train'), 
            ans[:len(train_data)]
            )
    np.save(
            os.path.join(config.ans_dir, 'y_valid'),
            ans[len(train_data):len(train_data) + len(valid_data)]
            )
    np.save(
            os.path.join(config.ans_dir, 'y_test'),
            ans[len(train_data) + len(valid_data):]
            )

def main():
    if not os.path.exists(config.obj_dir):
        os.mkdir(config.obj_dir)

    # loads raw data
    print('start loading raw data')
    train_data = json.load(open(os.path.join(config.data_dir, 'train.json')))
    valid_data = json.load(open(os.path.join(config.data_dir, 'valid.json')))
    test_data = json.load(open(os.path.join(config.data_dir, 'test.json')))

    # extracts ids and answers
    # extract_ids_ans(train_data, valid_data, test_data, config)

    # extracts feature
    # extract_base_feature(train_data, valid_data, test_data, config)

    # extract semantic feature
    extract_semantic_feature(train_data, valid_data, test_data, config)

if __name__ == '__main__':
    main()
