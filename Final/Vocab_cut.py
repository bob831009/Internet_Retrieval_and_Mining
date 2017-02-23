import json

# fp = open('vocab.json'); 
# data = json.load(fp);
# new_data = {};
# for key in data.keys():
# 	if(data[key] >= 50):
# 		new_data[key] = data[key];
# fp_output = open('vocab_cut.json', 'w');
# json.dump(new_data, fp_output);


fp = open('vocab_emotion.json'); 
data = json.load(fp);
new_data = {};
for key in data.keys():
	if(data[key] < -0.5):
		print "%s: %lf" % (key.encode('utf-8'), data[key]);
		