from sklearn.decomposition import LatentDirichletAllocation
import os
import json
import numpy as np

data_dir = ['train.json', 'valid.json'];
data_class = ['train', 'valid', 'test'];

Vocab_cut = json.load(open('vocab_cut.json'));
i = 0;
Vocab_to_index = {};
for key in Vocab_cut.keys():
	Vocab_to_index[key] = i;
	i += 1;

clf = LatentDirichletAllocation(batch_size=1024, n_topics=20);
handle_doc_num = 0;

for file_name in data_dir:
	print("Handling %s LDA" % file_name);
	Data = json.load(open(file_name));
	handle_doc_num = 0;
	Tmp_Total_Metrix = [];
	for Doc_obj in Data:
		Tmp_metrix = [0] * len(Vocab_cut);
		handle_doc_num += 1;
		Doc_Content = Doc_obj['main_content'].split(" ");
		for term in Doc_Content:
			if(term in Vocab_to_index):
				Tmp_metrix[Vocab_to_index[term]] += 1;

		Tmp_Total_Metrix.append(Tmp_metrix);

		if(handle_doc_num >= 1024):
			clf.partial_fit(Tmp_Total_Metrix);
			Tmp_Total_Metrix = [];
			handle_doc_num = 0;

data_dir.append('test.json');
for i in range(3):
	print("Predicting %s By LDA Model" % data_class[i]);
	file_name = data_dir[i];
	Data = json.load(open(file_name));
	output_data = [];
	for Doc_obj in Data:
		Tmp_metrix = [0] * len(Vocab_cut);
		Doc_Content = Doc_obj['main_content'].split(" ");
		for term in Doc_Content:
			if(term in Vocab_to_index):
				Tmp_metrix[Vocab_to_index[term]] += 1;
		transform_metrix = clf.transform([Tmp_metrix]);
		output_data.append(transform_metrix);

	output_data = np.array(output_data);
	np.save(os.path.join('/tmp2', 'b02902015', 'ir', 'obj', 'feature', data_class[i]+'_LDA'), output_data);
