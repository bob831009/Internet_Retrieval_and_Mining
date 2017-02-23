import os
import re
import math
import sys;
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer();
LAMDA = 0.8;
def Smooth(tmp, LAMDA, Word_to_index, back_ground_prob):
	for i in range(len(tmp)):
		tmp[i] = math.log(LAMDA*tmp[i] + (1-LAMDA)*back_ground_prob[i]);
	return tmp;
def GetVocabNum(file_path, Word_to_index, tmp):
	f = open(file_path, "r", encoding = "utf-8", errors = "ignore");
	for line in f:
		line = " ".join(re.findall("[a-zA-Z]+", line));
		line = line.strip().split(" ");
		for word in line:
			if(word == ''):
				continue;
			word = stemmer.stem(word.lower());
			if(word in Word_to_index):
				tmp[Word_to_index[word]] += 1;
	return tmp;
def ConstructModel(train_dir, Total_category, Word_to_index, back_ground_prob, Label_Size):
	Model = np.array([]);
	Model.shape = (0, len(Word_to_index));
	for category in Total_category:
		tmp = np.zeros(len(Word_to_index.keys()));
		tmp_label_doc_num = 0;
		for file_name in os.listdir(os.path.join(train_dir, category)):
			file_path = os.path.join(train_dir, category, file_name);
			if(tmp_label_doc_num < Label_Size or Label_Size == -1):
				tmp = GetVocabNum(file_path, Word_to_index, tmp);
				tmp_label_doc_num += 1;
			if(tmp_label_doc_num >= Label_Size and Label_Size != -1):
				break;
		tmp /= sum(tmp);
		tmp = Smooth(tmp, LAMDA, Word_to_index, back_ground_prob);
		tmp.shape = (1, len(Word_to_index));
		Model = np.concatenate((Model, tmp), axis=0);
	return Model;

def AddWordNumFromDoc(file_path, Total_Vocab):
	f = open(file_path, "r", encoding = "utf-8", errors = "ignore");
	for line in f:
		line = " ".join(re.findall("[a-zA-Z]+", line));
		line = line.strip().split(" ");
		for word in line:
			if(word == ''):
				continue;
			word = stemmer.stem(word.lower());
			if(word not in Total_Vocab):
				Total_Vocab[word] = 1;
			else:
				Total_Vocab[word] += 1;

	return Total_Vocab;

def ConstructVocab(input_dir, source, Label_Size):
	Total_category = [];
	Total_Vocab = {};
	Total_category_num = [];
	for source in Source:
		if(source == 'Train'):
			for category in os.listdir(os.path.join(input_dir, source)):
				if(category == '.DS_Store'):
					continue;
				Total_category.append(category);
				tmp_label_doc_num = 0;
				for file_name in os.listdir(os.path.join(input_dir, source, category)):
					file_path = os.path.join(input_dir, source, category, file_name);
					if(tmp_label_doc_num < Label_Size or Label_Size == -1):
						AddWordNumFromDoc(file_path, Total_Vocab);
						tmp_label_doc_num += 1;
					if(tmp_label_doc_num >= Label_Size and Label_Size != -1):
						break;
				Total_category_num.append(float(tmp_label_doc_num));

		else:
			for file_name in os.listdir(os.path.join(input_dir, source)):
				file_path = os.path.join(input_dir, source, file_name);
				Total_Vocab = AddWordNumFromDoc(file_path, Total_Vocab);
	return Total_category, Total_Vocab, Total_category_num;
def predict_test(test_dir, Model, Word_to_index, Total_category, category_prob):
	Prediction = {};
	Prediction_name = [];
	for file_name in os.listdir(test_dir):
		# print "Handling " + file_name;
		file_path = os.path.join(test_dir, file_name);
		tmp = [0] * len(Word_to_index);
		tmp = GetVocabNum(file_path, Word_to_index, tmp);
		tmp = np.array(tmp);
		ans = tmp.dot(np.array(Model).T);
		ans = ans + np.log(category_prob);
		Max_index = np.argmax(ans);
		Prediction[file_name] = Total_category[Max_index];
		Prediction_name.append(int(file_name));
		# print Total_category[Max_index];
	return Prediction, Prediction_name;
def Parse_Argv(argv):
	argc = len(argv);
	Label_Size = -1;
	if(argc < 5 or argc > 7):
		print ("ERROR ARGV FORMAT!")
		sys.exit();
	for i in range(1, argc):
		if(argv[i] == '-i'):
			input_dir = argv[i+1];
		elif(argv[i] == '-o'):
			output_dir = argv[i+1];
		elif(argv[i] == '-n'):
			Label_Size = int(argv[i+1]);
	return input_dir, output_dir, Label_Size;
	
Source = ['Train', 'Unlabel'];
Ans_dir = 'test_ans'
if __name__ == '__main__':
	print ("Parse_Argv");
	input_dir, output_dir, Label_Size = Parse_Argv(sys.argv);
	print ("Construct Vocab")
	Total_category, Total_Vocab, Total_category_num = ConstructVocab(input_dir, Source, Label_Size);
	Word_to_index = {};
	Total_Vocab_list = np.zeros(len(Total_Vocab.keys()));
	i = 0;
	for word in Total_Vocab.keys():
		Word_to_index[word] = i;
		Total_Vocab_list[i] = Total_Vocab[word];
		i += 1;
	# print Word_to_index;
	train_dir = os.path.join(input_dir, 'Train');
	print ("Construct back_ground_prob")
	back_ground_prob = Total_Vocab_list/sum(Total_Vocab_list);

	print ("Construct category prob");
	Total_category_num = np.array(Total_category_num);
	category_prob = Total_category_num/sum(Total_category_num);
	print ("Construct Model")
	Model = ConstructModel(train_dir, Total_category, Word_to_index, back_ground_prob, Label_Size);

	print ("Start Prediction")
	test_dir = os.path.join(input_dir, 'Test');
	Prediction, Prediction_name = predict_test(test_dir, Model, Word_to_index, Total_category, category_prob);


	# Correct = 0;
	# Total_test_case = 0;
	# f = open(Ans_dir, "r");
	# for line in f:
	# 	Total_test_case += 1;
	# 	line = line.strip().split(" ");
	# 	if(Prediction[line[0]] == line[1]):
	# 		Correct += 1;
	# print "Accuary: %lf" % (float(Correct) / Total_test_case);

	Prediction_name.sort();
	print ("Output to file");
	f = open(output_dir, "w");
	for i in range(len(Prediction)):
		f.write("%s %s\n" % (Prediction_name[i], Prediction[str(Prediction_name[i])]));