import json
import numpy
import math

def Handle_range(Score):
	if(Score < -100):
		return -110;
	if(Score > 100):
		return 100;

	tmp_range = math.floor(Score/10)*10;
	return tmp_range;

train_data = json.load(open("train.json"));
valid_data = json.load(open("valid.json"));

Score_range = {};
for i in range(-110, 110, 10):
	Score_range[i] = 0;

for Doc_obj in (train_data + valid_data):
	Score = Doc_obj['messageNum']['g'] - Doc_obj['messageNum']['b'];
	tmp_Score_range = Handle_range(Score);
	Score_range[tmp_Score_range] += 1;

for key in Score_range.keys():
	print("%d, %d" % (key, Score_range[key]));