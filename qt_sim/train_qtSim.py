#! -*- coding:utf-8 -*-
# qt spam任务训练

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
from keras.layers import Dropout, Dense
from keras.utils import multi_gpu_model
from tensorflow.python.keras import backend as KTF
import keras.backend.tensorflow_backend as tfback

set_gelu('tanh')  # 切换gelu版本

maxlen = 128
batch_size = 128
epochs = 2
use_gpu = ''
gpus = 0
pretrained_path = '../models/chinese_simbert_L-12_H-768_A-12'
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
			label = data_list[2]
			D.append((query, title, int(label)))
	return D

datapath = '../data/'
# 加载数据集
train_data = load_data(datapath + 'test_data.txt')
valid_data = load_data(datapath + 'test_data.txt')
test_data = load_data(datapath + 'test_data.txt')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
	"""数据生成器
	"""
	def __iter__(self, random=False):
		batch_token_ids, batch_segment_ids, batch_labels = [], [], []
		for is_end, (query, title, label) in self.sample(random):
			token_ids, segment_ids = tokenizer.encode(
				query, title, max_length=maxlen) #maxlen = maxlen
			batch_token_ids.append(token_ids)
			batch_segment_ids.append(segment_ids)
			batch_labels.append([label])
			if len(batch_token_ids) == self.batch_size or is_end:
				batch_token_ids = sequence_padding(batch_token_ids)
				batch_segment_ids = sequence_padding(batch_segment_ids)
				batch_labels = sequence_padding(batch_labels)
				yield [batch_token_ids, batch_segment_ids], batch_labels
				batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert1 = build_transformer_model(
	config_path=config_path,
	checkpoint_path=checkpoint_path,
	with_pool='linear',
	return_keras_model=False,
)

bert2 = build_transformer_model(
	config_path=config_path,
	checkpoint_path=checkpoint_path,
	with_pool=False,
	return_keras_model=False,
)
print("--->bert1 model output:",bert1.output)
print("--->bert2 model output:",bert2.output)
for layer in bert2.layers:
	print(layer.name)

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
			print("Save Model Succussfully!", mf)
		test_acc = evaluate(test_generator)
		print('--->val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
			(val_acc, self.best_val_acc, test_acc))

if __name__ == '__main__':
	train = False
	mf = "qt_sim_model.h5"
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

	evaluator = Evaluator()

	strategy = tf.distribute.MirroredStrategy(devices=devices)
	with strategy.scope():
		output = Dropout(rate=0.1)(bert1.model.output)
		# output = Dense(units=2, activation='softmax', kernel_initializer=bert.initializer)(output)
		output = Dense(units=3, activation='softmax', kernel_initializer=bert1.initializer)(output)
		model = keras.models.Model(bert1.model.input, output)
		if gpus > 0:
			model = multi_gpu_model(model,gpus=gpus)
		# model.summary()

		model.compile(
			loss='sparse_categorical_crossentropy',
			optimizer=Adam(2e-5),  # 用足够小的学习率
			# optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
			metrics=['sparse_categorical_accuracy'],
		)
	# 写好函数后，启用对抗训练只需要一行代码
	#adversarial_training(model, 'Embedding-Token', 0.5)

	if train:
		model.fit(
			train_generator.forfit(),
			steps_per_epoch=len(train_generator),
			epochs=epochs,
			callbacks=[evaluator]
		)

		print(u'final test acc: %05f\n' % (evaluate(test_generator)))

