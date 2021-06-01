#simbert 非中心词遍历pred
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import initializers
from keras import regularizers
from sklearn.metrics import r2_score  # 用于评估模型
from sklearn.metrics import mean_squared_error  # 用于评估模型
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer,load_vocab
from bert4keras.snippets import to_array
from bert4keras.backend import keras
from bert4keras.snippets import sequence_padding
import keras.backend.tensorflow_backend as KTF


os.environ["CUDA_VISIBLE_DEVICES"]=''

pretrained_path = 'models/chinese_simbert_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
dict_path = os.path.join(pretrained_path, 'vocab.txt')

token_dict = load_vocab(dict_path)
new_token_dict = token_dict.copy()
tokenizer = Tokenizer(new_token_dict, do_lower_case=True) 

batch_size = 10000
maxlen = 128
model_path = 'qt_sim_model/qt_sim_model_01_0.0261.h5'
model = keras.models.load_model(model_path)
def clean_term(term):
	return term[0:term.rfind("|")]

#数据集路径
test_data = 'data/tmp.txt' # 'test1w.txt' #'/home/data/qt_spam/topic_qt_sample.txt'
pred_output_path = 'predict/predict_tmp.csv'
if __name__=='__main__':
	infopen = open(test_data,'r')
	outopen = open(pred_output_path,'w')
	lines = infopen.readlines()
	test_sample_num = 0
	same_label_num = 0
	i = 0
	batch_token_ids = []
	batch_segment_ids = []
	for line in lines:
		test_sample_num += 1
		line_list = line.strip().split("\t")
		query = line_list[0]
		title = line_list[1]
		label = line_list[2]
		token_ids, segment_ids = tokenizer.encode(query,title,maxlen = maxlen)
		batch_token_ids.append(token_ids)
		batch_segment_ids.append(segment_ids)
		i += 1
		if i <= batch_size:
			token_list = sequence_padding(batch_token_ids)
			segment_list = sequence_padding(batch_segment_ids)
			batch_token_ids = []
			batch_segment_ids = []

			model_predict = model.predict([token_list,segment_list]) # .argmax(axis=1)
			print('model_predict:', model_predict)
			sim = model_predict[0][0]
			out_line = '{}\t{}\t{}\t{}\n'.format(query, title, sim, label)
			outopen.write(out_line + "\n")

	infopen.close()
	outopen.close()
	print('---->ok saved in {}'.format(pred_output_path))
	print('---->predict completed')

