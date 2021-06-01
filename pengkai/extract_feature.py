#！-*- coding: utf-8 -*-

import os
import pdb
import time
import datetime
import json
import math
import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import uniout, open
from keras.models import Model

maxlen = 100
batch_size = 1000

model_dir = '/dmcv1/wangpengkai/pretrained_models/nlp/simbert/chinese_simbert_L-12_H-768_A-12'
raw_data_file = '/dmcv1/wangpengkai/datasets/nlp/autohome/query_question_recall/query_2_query_0914.csv'
train_data_feature_file = '/dmcv1/wangpengkai/datasets/nlp/autohome/query_question_recall/bert/query_data_feature/query_2_query_0914_feature.csv'
#valid_data = load_test_data('/dmcv1/wangpengkai/datasets/nlp/autohome/query_question_recall/query_2_query_0914.csv')

model_dir = sys.stdin[1]
raw_data_file = sys.stdin[2]
train_data_feature_file = sys.stdin[3]

# bert配置
config_path = os.path.join(model_dir, 'bert_config.json')
checkpoint_path = os.path.join(model_dir, 'bert_model.ckpt')
dict_path = os.path.join(model_dir, 'vocab.txt')

#config_path = '/dmcv1/wangpengkai/pretrained_models/nlp/simbert/chinese_simbert_L-6_H-384_A-12/bert_config.json'
#checkpoint_path = '/dmcv1/wangpengkai/pretrained_models/nlp/simbert/chinese_simbert_L-6_H-384_A-12/bert_model.ckpt'
#dict_path = '/dmcv1/wangpengkai/pretrained_models/nlp/simbert/chinese_simbert_L-6_H-384_A-12/vocab.txt'

#config_path = '/dmcv1/wangpengkai/pretrained_models/nlp/simbert/chinese_simbert_L-4_H-312_A-12/bert_config.json'
#checkpoint_path = '/dmcv1/wangpengkai/pretrained_models/nlp/simbert/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
#dict_path = '/dmcv1/wangpengkai/pretrained_models/nlp/simbert/chinese_simbert_L-4_H-312_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for index, l in enumerate(f):
            try:
                l_split = l.strip().split('\t')
                if len(l_split) != 18:
                    continue

                object_uid = l_split[0]
                long_title = l_split[2]
                info_obj = l_split[3]
                if long_title in ['NULL', '']:
                    continue
                D.append((index, object_uid, long_title, info_obj))
            except Exception as e:
                print('load data Exception:{}'.format(e))
                continue
    return D

def load_test_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l_split = l.strip()
            if l_split == '':
              continue
            D.append(l_split)
    return D

start_time = time.time()
print('start time:{}'.format(datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d %H:%M:%S")))

# 测试相似度效果
a_token_ids, object_uids, index_ids, long_titles = [], [], [], []
texts = []

#train_data_feature_file = '/dmcv1/wangpengkai/datasets/nlp/autohome/query_question_recall/bert/query_data_feature/query_2_query_0914_feature.csv'

#pdb.set_trace()
if os.path.exists(train_data_feature_file):
  print('load feature')
  vecs = []
  with open(train_data_feature_file, 'r', encoding='utf-8') as f:
    for line in f:
      line_split = line.strip().split('\t')
      long_title = line_split[0]
      text_vec = json.loads(line_split[1])
      long_titles.append(long_title)
      vecs.append(text_vec)

  vecs = np.asarray(vecs)
else:
  print('extract feature')
  # 加载数据集
  #train_data = load_data('/dmcv1/wangpengkai/datasets/nlp/autohome/query_question_recall/res_deduplicate.dat')
  #train_data = load_test_data('/dmcv1/wangpengkai/datasets/nlp/autohome/query_question_recall/query.dat')
  train_data = load_test_data(raw_data_file)
  valid_data = load_test_data(raw_data_file)
  test_data = load_test_data(raw_data_file)

  train_data_feature = open(train_data_feature_file, 'w', encoding='utf-8')
  
  total_count = len(train_data)
  num_steps = int(math.ceil(total_count / batch_size))
  print('num_steps:{}'.format(num_steps))
  
  def get_batch(data_args, batch_count):
    start = batch_count * batch_size
    end = (batch_count + 1) * batch_size
  
    batch_data = data_args[start:end]
  
    return batch_data
  
  for step in range(num_steps):
      print('step:{}'.format(step))
      try:
          batch_data = get_batch(train_data, step)
  
          for i, d in enumerate(batch_data):
              long_title = d
              #index = d[0]
              #object_uid = d[1]
              #long_title = d[2]
              #info_obj = d[3]
  
              token_ids = tokenizer.encode(long_title, max_length=maxlen)[0]
              a_token_ids.append(token_ids)
              #object_uids.append(object_uid)
              #index_ids.append(index)
              long_titles.append(long_title)
              texts.append(long_title)
  
          a_token_ids_arr = sequence_padding(a_token_ids)
          a_vecs = encoder.predict([a_token_ids_arr, np.zeros_like(a_token_ids_arr)],
                                   verbose=True)
  
          a_vecs = a_vecs / (a_vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
          for j, a_vec in enumerate(a_vecs):
              train_data_feature.write(long_titles[j] + '\t' + json.dumps(a_vec.tolist()) + '\n')
  
          # pdb.set_trace()
          if step == 0:
              vecs = a_vecs
          else:
              # pdb.set_trace()
              vecs = np.concatenate([vecs, a_vecs], axis=0)
              # vecs = np.concatenate([vecs, a_vecs], axis=1).reshape(-1, 768)
          a_token_ids, object_uids, index_ids, long_titles = [], [], [], []
  
      except Exception as e:
          print('exception:{}'.format(e))
          continue
      finally:
          a_token_ids, object_uids, index_ids, long_titles = [], [], [], []
  
  train_data_feature.close()
  
  end_time = time.time()
  print('train end time:{}'.format(end_time - start_time))
  print('train end time:{}'.format(datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d %H:%M:%S")))

def most_similar(text, topn=10):
    """检索最相近的topn个句子
    """
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    vec = encoder.predict([[token_ids], [segment_ids]])[0]
    vec /= (vec**2).sum()**0.5
    sims = np.dot(vecs, vec)
    return [(texts[i], sims[i]) for i in sims.argsort()[::-1][:topn]]

#valid_result_file = open('/dmcv1/wangpengkai/datasets/nlp/autohome/query_question_recall/bert_384/query_data_feature/res_deduplicate_result_temp.dat', 'w', encoding='utf-8')
#
#for d in valid_data:
#  #pdb.set_trace()
#  try:
#      valid_result_file.write(d + '\t' + str(most_similar(d, 50)) + '\n')
#  except Exception as e:
#      print('valid exception:{}'.format(e))
#      continue
#
#valid_result_file.close()

end_time = time.time()
print('end time:{}'.format(end_time - start_time))
print('end time:{}'.format(datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d %H:%M:%S")))
