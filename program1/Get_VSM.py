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
tmp_line_num = 0;
row = [];
col = [];
data = [];
data_tf = [];
data_w = [];
Total_IDF = [];
f = open(model_dir + "/inverted-file", "r");
fp_term_all = open("./term.txt", "w");
for line in f:
	line = line.strip().split(" ");
	line = list(map(int, line));
	tmp_term = str(line[0])+" "+str(line[1]);

	# if(line[0] in StopWord_index or line[1] in StopWord_index or (len(Vocab[line[0]]) == 1 and line[1] == -1)):
	# 	for i in range(line[2]):
	# 		f.readline();
	# 	continue;
	fp_term_all.write("%s\n" % tmp_term);
	Vocab_to_Term[tmp_term] = tmp_line_num;
	tmp_idf = math.log(float(Total_doc_num)/line[2]);
	Total_IDF.append(tmp_idf);

	for i in range(line[2]):
		new_line = f.readline();
		new_line = new_line.strip().split(" ");
		new_line = list(map(int, new_line));
		
		row.append(new_line[0]);
		col.append(tmp_line_num);
		data.append(new_line[1]);
		row_tf = float(new_line[1])/Doc_len[new_line[0]];
		k = 0.75;
		b = 0.75;
		norm_tf = (k + 1) * new_line[1] / (new_line[1] + k*(1 - b + b * float(Doc_len[new_line[0]])/Doc_len_mean));
		data_tf.append(norm_tf);
		data_w.append(norm_tf * tmp_idf);
	
	tmp_line_num += 1;
Total_term_num = tmp_line_num;

Total_W = csr_matrix((data_w, (row, col)), shape=(Total_doc_num, Total_term_num));
save_sparse_csr("VSM", Total_W);