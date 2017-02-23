import json

def HandleMessageTag(tag):
	if(tag == u'\u63a8 '):
		return 1;
	elif(tag == u'\u5653 '):
		return -1;
	else:
		return 0;

Vocab_Emotion = {};
Vocab_Emotion_Num = {};

Vocab_file = open("vocab_cut.json");
Vocab_data = json.load(Vocab_file);
for key in Vocab_data.keys():
	Vocab_Emotion[key] = 0;
	Vocab_Emotion_Num[key] = 0;

input_fp = open('train.json'); 
data = json.load(input_fp);
for elem in data:
	content = elem;
	# print "==========";
	# print content;
	for message_id in content['message'].keys():
		message_obj = content['message'][message_id];
		message_content = message_obj['push_content'];
		message_content = message_content.strip().split(" ");
		# print message_obj['push_tag'];
		Score = HandleMessageTag(message_obj['push_tag']);
		# print Score;
		for word in message_content:
			if(word in Vocab_Emotion):
				Vocab_Emotion[word] += Score;
				Vocab_Emotion_Num[word] += 1;

for key in Vocab_Emotion.keys():
	if(Vocab_Emotion_Num[key] > 0):
		Vocab_Emotion[key] = float(Vocab_Emotion[key]) / Vocab_Emotion_Num[key];

output_fp = open('vocab_emotion.json', 'w');
json.dump(Vocab_Emotion, output_fp);