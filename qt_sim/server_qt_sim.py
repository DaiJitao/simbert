#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import requests
import tornado.ioloop
import tornado.web
import tornado.escape
import threading
import time
from tornado import gen
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
import logging
import json
import math
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer,load_vocab
from bert4keras.backend import keras
from bert4keras.snippets import sequence_padding
import collections
import tensorflow as tf

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
model_path = 'resources/rescoure/qt_sim_model_01_0.0261.h5'
model = keras.models.load_model(model_path)

pretrained_path = 'resources/rescoure/chinese_simbert_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
dict_path = os.path.join(pretrained_path, 'vocab.txt')
token_dict = load_vocab(dict_path)
new_token_dict = token_dict.copy()
tokenizer = Tokenizer(new_token_dict, do_lower_case=True)

maxlen = 128

class MainHandler(tornado.web.RequestHandler):
	executor = ThreadPoolExecutor(32)

	def post(self):
		data = {}
		data['code'] = "only get allowed"
		self.set_header('Content-Type', 'application/json')
		self.write(tornado.escape.json_encode(data))
		logger.info(data)
		self.set_header("Content-Type", "application/json")
		self.write(tornado.escape.json_encode(data))
	
	@gen.coroutine
	def get(self):
		data = {}
		try:
			query = self.get_query_argument('query')
			title = self.get_query_argument('title')
			logger.info('query -> {},title -> {}'.format(query,title))
			if query is not None and title is not None:
				query = query.strip()
				title = title.strip()
				token_ids, segment_ids = tokenizer.encode(query, title.replace(' ',','), maxlen = maxlen)
				token_list = sequence_padding([token_ids])
				segment_list = sequence_padding([segment_ids])
				model_predict = model.predict([token_list,segment_list]) # .argmax(axis=1)
				sim = model_predict[0][0]
				if sim <=0.5:
					label = '0'
				elif sim > 0.5 and sim <= 1.5:
					label = '1'
				else:
					label = '2'
				data['result'] = str(sim)
				data['label'] = label
			else:
				data["returnCode"] = 1
		except Exception as e:
			data["returnCode"] = 1
			logger.info("error")
		logger.info(data)
		self.set_header("Content-Type", "application/json")
		self.write(tornado.escape.json_encode(data))

	@run_on_executor
	def sleep(self):
		time.sleep(50)
		return 50


application = tornado.web.Application([
	#(r"/",MainHandler),
	(r"/qt/sim",MainHandler),
])

if __name__ == "__main__":
	#设置显存分配比例
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.333
	session = tf.compat.v1.Session(config=config)
	tf.compat.v1.keras.backend.set_session(session)

	application.listen(85001)
	tornado.ioloop.IOLoop.instance().start()
