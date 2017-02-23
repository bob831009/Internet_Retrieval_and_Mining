import xml.etree.ElementTree as ET
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import math
import sys

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

print ("Count Doc len");
Doc = [];
f = open(model_dir + "/file-list", "r");
fp_output = open("./Doc_len.txt", "w");
for line in f:
	doc_path = line.strip();
	Doc.append(doc_path);
	new_dir = NTCR_dir + "/" + doc_path.split("/", 1)[1];
	fp_output.write("%d\n" % XML_parser_getLen(new_dir));