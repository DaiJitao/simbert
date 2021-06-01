#! -*- coding:utf-8 -*-
# qt sim 任务训练:冻结部分层数，最后做cosin
import sys
import os
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras
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
mf = "model/qt_sim_model.h5"
maxlen = 128
batch_size = 384  # 256
epochs = 10
use_gpu = ''  # '0,1,2,3'
gpus = 0  # 4
if isLinux:
    pretrained_path = 'models/chinese_simbert_L-6_H-384_A-12'  # 'models/chinese_simbert_L-12_H-768_A-12'
else:
    pretrained_path = '../models/chinese_simbert_L-4_H-312_A-12'  # 'models/chinese_simbert_L-12_H-768_A-12'

config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
dict_path = os.path.join(pretrained_path, 'vocab.txt')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    # global _LOCAL_DEVICES
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


def convertPredictData(sentence_a, sentence_b):
    query = sentence_a.strip()
    title = sentence_b.strip()
    batch_token_ids, batch_segment_ids, batch_token_ids2, batch_segment_ids2 = [], [], [], []
    token_ids, segment_ids = tokenizer.encode(query, maxlen=maxlen)
    token_ids2, segment_ids2 = tokenizer.encode(title, maxlen=maxlen)
    batch_token_ids.append(token_ids)
    batch_segment_ids.append(segment_ids)
    batch_token_ids2.append(token_ids2)
    batch_segment_ids2.append(segment_ids2)
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_token_ids2 = sequence_padding(batch_token_ids2)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    batch_segment_ids2 = sequence_padding(batch_segment_ids2)
    data = [batch_token_ids, batch_segment_ids, batch_token_ids2, batch_segment_ids2]
    return data

if __name__ == '__main__':
    tfback._get_available_gpus = _get_available_gpus
    # 自适应分配显存
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    #KTF.set_session(session)

    #modelf = 'qt_sim_models/qtSim_twinNetword_05_0.995.h5'
    modelf = sys.argv[1].strip()
    print('used model:' + modelf )
    myModel = keras.models.load_model(modelf, {'softmax_v2': tf.nn.softmax})
    myModel.summary()
    for index, layer in enumerate(myModel.layers):
        print(index, layer, layer.name)
        if index == 20:
            print(dir(layer))
            print('output_names:', layer.output_names)
            print('output_shape:', layer.output_shape)
            print('outputs:', layer.outputs)
            for j, innerLayer in enumerate(layer.layers):
                print(j, innerLayer, innerLayer.output, innerLayer.name)

    twinbertModelmyModel = myModel.get_layer(index=20)
    print('twinbertModelmyModel:')
    twinbertModelmyModel.summary()
    query_vec = twinbertModelmyModel.get_layer('q-global-avg-pool').output
    title_vec = twinbertModelmyModel.get_layer('t-global-avg-pool').output
    print('query_vec:', query_vec, ' , title_vec:',title_vec)
    if True:
        inf,outf=sys.argv[2].strip(), sys.argv[3].strip()
        outfp = open(outf, encoding='utf-8', mode='w')
        with open(inf, encoding='utf-8', mode='r') as fp:
            for line in fp:
                arr = line.strip().split('\t')
                q,t,label = arr[0], arr[1], int(arr[3].strip())
                query = q
                title = t
                data = convertPredictData(query, title)
                sim = twinbertModelmyModel.predict(data)[0][0] #1*1
                temp = '{}\t{}\t{}\t{}\n'.format(q,t,label, sim)
                outfp.write(temp)
        outfp.close()
        print('ok saved in {}'.format(outf))

        #res = myModel.predict(data)
        #print('qyery:', query)
        #print('title:', title)
        #print(res.shape)
        #print(type(res))
        #print(res)

