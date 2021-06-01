#! -*- coding:utf-8 -*-
# qt sim 任务训练:冻结部分层数，最后做cosin

import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from bert4keras.backend import keras, set_gelu, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense, Lambda, Concatenate, Reshape, AveragePooling1D
from keras.utils import multi_gpu_model
from tensorflow.python.keras import backend as KTF
import keras.backend.tensorflow_backend as tfback
from keras.callbacks import ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = ""
isLinux = True
set_gelu('tanh')  # 切换gelu版本
mf="model/qt_sim_model.h5"
maxlen = 128
batch_size = 384 #256
epochs = 10
use_gpu = '' # '0,1,2,3'
gpus = 0 #4
if isLinux:
	pretrained_path = 'models/chinese_simbert_L-4_H-312_A-12'  # 'models/chinese_simbert_L-12_H-768_A-12'
else:
	pretrained_path = '../models/chinese_simbert_L-4_H-312_A-12'  # 'models/chinese_simbert_L-12_H-768_A-12'

config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
dict_path = os.path.join(pretrained_path, 'vocab.txt')

def _get_available_gpus():
	"""Get a list of available gpu devices (formatted as strings).

	# Returns
		A list of available GPU devices.
	"""
	#global _LOCAL_DEVICES
	if tfback._LOCAL_DEVICES is None:
		devices = tf.config.list_logical_devices()
		tfback._LOCAL_DEVICES = [x.name for x in devices]
	return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

def load_data(filename):
	"""加载数据
	单条格式：(query, title, label)
	"""
	D = []
	with open(filename, encoding='utf-8') as f:
		for l in f:
			data_list = l.strip().split('\t')
			if not len(data_list) == 3:
				continue
			query = data_list[0]
			title = data_list[1]
			label = int(data_list[2])
			D.append((query, title, label))
			# D.append((title, query, label))
	return D

if isLinux:
	datapath = 'data/'
else:
	datapath = '../data/'
# 加载数据集
train_data = load_data(datapath + 'train_data_shuf300w.txt')
valid_data = load_data(datapath + 'validate_data.txt')
test_data = load_data(datapath + 'test_data_1bai.txt')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
	"""数据生成器
	"""
	def __iter__(self, random=False):
		batch_token_ids, batch_segment_ids, batch_labels = [], [], [] # 针对句子A
		batch_token_ids2, batch_segment_ids2 = [], [] #针对句子B
		for is_end, (query, title, label) in self.sample(random):
			token_ids, segment_ids = tokenizer.encode(query, maxlen=maxlen)
			batch_token_ids.append(token_ids)
			batch_segment_ids.append(segment_ids)
			batch_labels.append([label])
			token_ids2, segment_ids2 = tokenizer.encode(title, maxlen= maxlen)
			batch_token_ids2.append(token_ids2)
			batch_segment_ids2.append(segment_ids2)
			if len(batch_token_ids) == self.batch_size or is_end:
				batch_token_ids = sequence_padding(batch_token_ids)
				batch_token_ids2 = sequence_padding(batch_token_ids2)

				batch_segment_ids = sequence_padding(batch_segment_ids)
				batch_segment_ids2 = sequence_padding(batch_segment_ids2)

				batch_labels = sequence_padding(batch_labels)
				yield [batch_token_ids, batch_segment_ids, batch_token_ids2, batch_segment_ids2], batch_labels
				batch_token_ids, batch_segment_ids, batch_labels = [], [], []
				batch_token_ids2, batch_segment_ids2 = [], []

# 加载预训练模型
bert = build_transformer_model(
	config_path=config_path,
	checkpoint_path=checkpoint_path,
	with_pool=True, #with_nsp=False, with_mlm=True,
	model='bert',
	application='encoder',
	return_keras_model=False,
)
# 修改模型的层名字，全局唯一
for index, layer in enumerate(bert.model.layers):
	temp = layer.name + '-model1'
	layer.name = temp # print(index, layer.name, bert.model.get_layer(index=index).name)

bert2 = build_transformer_model(
	config_path=config_path,
	checkpoint_path=checkpoint_path,
	with_pool=True, #with_nsp=False, with_mlm=True,
	model='bert',
	application='encoder',
	return_keras_model=False,
)
for index, layer in enumerate(bert2.model.layers):
	temp = layer.name + '-model2'
	layer.name = temp

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

def evaluate(data):
	total, right = 0., 0.
	for x_true, y_true in data:
		y_pred = model.predict(x_true).argmax(axis=1)
		y_true = y_true[:, 0]
		total += len(y_true)
		right += (y_true == y_pred).sum()
	return right / total

def cosine_distance(inputs):
	'''
	计算余弦距离
	:param inputs:
	:return:
	'''
	x, y = inputs
	x, y = K.l2_normalize(x, axis=-1), K.l2_normalize(y, axis=1)
	res = x * y
	res = K.sum(res, axis=-1)
	return K.expand_dims(res)

def absDiff(inputs):
	a_vec, b_vec = inputs[0], inputs[1]
	return K.abs(a_vec - b_vec)

class Evaluator(keras.callbacks.Callback):
	"""评估与保存
	"""
	def __init__(self):
		self.best_val_acc = 0.

	def on_epoch_end(self, epoch, logs=None):
		val_acc = evaluate(valid_generator)
		if val_acc > self.best_val_acc:
			self.best_val_acc = val_acc
			#model.save_weights('best_model.weights')
			model.save(mf)
			print("-->Save Model Successfully!" + mf)
		test_acc = evaluate(test_generator)
		print('-->val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
			(val_acc, self.best_val_acc, test_acc)
		)

if __name__ == '__main__':
	train = True
	savedir='qt_sim_models'
	modelname="qtSimNewData_twinNetword_{epoch:02d}_{sparse_categorical_accuracy:.3f}.h5"
	modelfile=os.path.join(savedir,modelname)

	tfback._get_available_gpus = _get_available_gpus
	devices = []
	for i in use_gpu.split(','):
		if i != '':
			devices.append('/gpu:{}'.format(i))
	print('---->gpu:', devices)
	#自适应分配显存
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth=True
	session = tf.compat.v1.Session(config=config)
	KTF.set_session(session)

	#evaluator = Evaluator()
	checkpoint = ModelCheckpoint(modelfile,monitor='sparse_categorical_accuracy',mode='max',verbose=1,period=1) # save_best_only=True
	strategy = tf.distribute.MirroredStrategy(devices=devices)
	with strategy.scope():
		print("-->bert model output:", bert.model.output) 	# shape=(None, 312)
		print("-->bert2 model output:", bert2.model.output) # shape=(None, 312)
		query_vec = Reshape([1, -1], name='query-reshape')(bert.model.output) 		# shape=(None, 1, hidden_size)
		title_vec = Reshape([1, -1], name='title-reshape')(bert2.model.output) 	# shape=(None, 1, hidden_size)
		# channels_last对应于具有形状(batch, length, channels)的输入，
		# channels_first对应于具有形状(batch, channels, length)的输入。
		query_avg = AveragePooling1D(pool_size=3,strides=2,padding='same',data_format='channels_first',name='avgpool_query')(query_vec)
		title_avg = AveragePooling1D(pool_size=3,strides=2,padding='same',data_format='channels_first',name='avgpool_title')(title_vec)
		query_vec = Reshape([-1], name='query-reshape-2')(query_avg) #(None, size)
		title_vec = Reshape([-1], name='title-reshape-2')(title_avg) #(None, size)
		abs_diff = Lambda(absDiff, name='abs-diff')([query_vec, title_vec])
		all_vec = Concatenate(axis=1, name='query-title-abs')([query_vec, title_vec, abs_diff])
		outputs = Dense(3, activation='softmax', kernel_initializer=bert.initializer, name='output')(all_vec)

		print('type:', type(bert.model.input+bert2.model.input))
		print(bert.model.input+bert2.model.input)
		model = keras.models.Model(inputs=bert.model.input+bert2.model.input, outputs=outputs)
		# 想要让某层参加训练，必须'先'让全部层[可训练]，'再'让不想参加训练的层[冻结].
		# model.trainable = True
		# 让不想参加训练的层[冻结]. 冻结前10 layer encode
		# for layer in model.layers[:88]:
		# 	layer.trainable = False

		if gpus > 0:
			model = multi_gpu_model(model,gpus=gpus)

		model.summary()
		#for i, layer in enumerate(model.layers):
			#print(i, layer, layer.name, layer.trainable)
		model.compile(
			loss='sparse_categorical_crossentropy',
			optimizer=Adam(2e-5),  # 用足够小的学习率
			# optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
			metrics=['sparse_categorical_accuracy'],
		)

	if train:
		model.fit(
			train_generator.forfit(),
			steps_per_epoch=len(train_generator),
			epochs=epochs,
			verbose=1,
			callbacks=[checkpoint]
		)
		print(u'--->final test acc: %05f\n' % (evaluate(test_generator)))
