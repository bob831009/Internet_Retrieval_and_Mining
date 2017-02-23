import xml.etree.ElementTree as ET
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import math
import sys

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def Find_term_index(input_str, Vocab_to_index, Vocab_to_Term):
	tmp_term_index = [];

	for i in range(len(input_str)):
		if(input_str[i] in Vocab_to_index):
			first_vocab_index = Vocab_to_index[input_str[i]];
			if((str(first_vocab_index) + " -1") in Vocab_to_Term):
				# print (Vocab_to_Term[str(first_vocab_index) + " -1"]);
				term_index = Vocab_to_Term[str(first_vocab_index) + " -1"];
				tmp_term_index.append(term_index);
			if(i + 1 < len(input_str) and input_str[i + 1] in Vocab_to_index):
				second_vocab_index = Vocab_to_index[input_str[i + 1]];
				if((str(first_vocab_index) + " " + str(second_vocab_index)) in Vocab_to_Term):
					term_index = Vocab_to_Term[str(first_vocab_index) + " " + str(second_vocab_index)];
					tmp_term_index.append(term_index);
	return tmp_term_index;

def XML_parser_getLen(path):
	Total_len = 0;
	tree = ET.parse(path);
	root = tree.getroot().find('doc');
	for title in root.findall('title'):
		if(title.text != None):
			Total_len += len(title.text);
		
	for text in root.findall('text'):
		for p in text.findall('p'):
			if(p.text != None):
				Total_len += len(p.text);

	return Total_len;
# handle option
feedback_control = 0;
for i in range(1, len(sys.argv)):
	if(sys.argv[i] == "-r"):
		feedback_control = 1;
		continue;
	elif(sys.argv[i] == "-i"):
		input_query_file = sys.argv[i + 1];
	elif(sys.argv[i] == "-o"):
		output_ranking_file = sys.argv[i + 1];
	elif(sys.argv[i] == "-m"):
		model_dir = sys.argv[i + 1];
	elif(sys.argv[i] == "-d"):
		NTCR_dir = sys.argv[i + 1];
	i += 1 ;


# handle Vocab
print ("handle Vocab");
Vocab_to_index = {};
Vocab = [];
f = open(model_dir + "/vocab.all", "r" , encoding = 'utf8');
tmp_line_num = 0;
for line in f:
	tmp_vocab = line.strip();
	Vocab_to_index[tmp_vocab] = tmp_line_num;
	Vocab.append(tmp_vocab);
	tmp_line_num += 1;
Total_vocab_num = tmp_line_num;

# Count Doc len
print ("Count Doc len");
Doc = [];
Doc_len = [];
tmp_line_num = 0;
f = open(model_dir + "/file-list", "r");
fp_Doc_len = open("./Doc_len.txt", "r");
for line in f:
	doc_path = line.strip();
	Doc.append(doc_path);
	tmp_len = int(fp_Doc_len.readline().strip());
	Doc_len.append(tmp_len);
	tmp_line_num += 1;
Total_doc_num = tmp_line_num;
Doc_len_mean = np.mean(np.array(Doc_len));


# handle stopword
print ("handle stopword");
StopWord_index = [];
f = open("./stoplist.zh_TW.u8", "r", encoding = 'utf8');
for line in f:
	tmp_stop_vocab = line.strip();
	StopWord_index.append(Vocab_to_index[tmp_stop_vocab]);


# Building Total_W
print ("building Total_W");
Vocab_to_Term = {};
f = open("./term.txt", "r");
tmp_line_num = 0;
for line in f:
	line = line.strip();
	Vocab_to_Term[line] = tmp_line_num;
	tmp_line_num += 1;
Total_term_num = tmp_line_num;
Total_W = load_sparse_csr("./VSM.npz");


# query train
test_output = open(output_ranking_file, "w");
tree = ET.parse(input_query_file);
# tree = ET.parse("./queries/query-train.xml");
root = tree.getroot();

print ("handle queries!");
for topic in root.findall('topic'):
	topic_len = 0;
	Topic_term_index = [];
	title = topic.find('title').text;
	topic_len += len(title);
	Topic_term_index.extend(Find_term_index(title, Vocab_to_index, Vocab_to_Term));

	Concepts = topic.find('concepts').text;
	topic_len += len(Concepts);
	Concepts = Concepts.strip().split("、");
	for concept in Concepts:
		Topic_term_index.extend(Find_term_index(concept, Vocab_to_index, Vocab_to_Term));
	
	topic_len += len(topic.find('narrative').text);

	# handle query_w;
	tmp_w = [0] * Total_term_num;
	for i in range(len(Topic_term_index)):
		tmp_w[Topic_term_index[i]] += 1;

	# sorting;
	score = [];
	Total_score = Total_W.dot(tmp_w);
	for i in range(Total_doc_num):
		tmp_score = Total_score[i];
		score.append([tmp_score, i]);
	score = sorted(score, key=lambda x: (x[0],x[1]), reverse=True);

	# feedback
	if(feedback_control == 1):
		for i in range(10):
			tmp_w = (np.array(tmp_w) + 0.5 * Total_W[score[i][1]])[0];
		tmp_w = np.array(tmp_w)[0]
		score = [];
		Total_score = Total_W.dot(tmp_w);
		for i in range(Total_doc_num):
			tmp_score = Total_score[i];
			score.append([tmp_score, i]);
		score = sorted(score, key=lambda x: (x[0],x[1]), reverse=True);


	# got topic index
	tmp_index = topic.find('number').text;
	topic_index = (tmp_index)[len(tmp_index)-3:];

	# output
	for i in range(100):
		test_output.write("%s %s\n" % (topic_index, (Doc[score[i][1]].split("/")[3]).lower()));
	