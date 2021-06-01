#! -*- coding:utf-8 -*-
# qt sim 任务训练:非冻结， 最后做cosin

import os
import sys
import tensorflow as tf
from keras.models import Sequential
from bert4keras.backend import keras, set_gelu, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense, Lambda, Input, Concatenate, BatchNormalization, Activation, Embedding, Reshape
from keras.utils import multi_gpu_model
from tensorflow.python.keras import backend as KTF
import keras.backend.tensorflow_backend as tfback
from keras.callbacks import ModelCheckpoint
from keras.models import Model

set_gelu('tanh')  # 切换gelu版本
# 测试集 训练集 验证集
trainf, testf, validatef = 'data/train_data.txt', 'data/test_data.txt', 'data/validate_data.txt'
test = False # 是否为测试环境
isUsegpu = True # 是否使用gpu
if isUsegpu:
    use_gpu = '0,1,2,3'
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu
    gpus = 4
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpus = 0
    use_gpu = ''

if test:
    modelfile = "model/qtSimSharedTwin_{epoch:02d}_{mae:.4f}.h5"
    config = {
        'maxlen': 128,
        'mf': "model/qt_sim_model.h5",
        'batch_size': 500,  # 256 批次大小
        'valid_length': 10000,  # 验证集总大小
        'epochs': 10,
        'use_gpu': use_gpu,
        'gpus': gpus,
        'pretrained_path': 'models/chinese_simbert_L-6_H-384_A-12',  # chinese_simbert_L-12_H-768_A-12',
        'train': True,
        'modelfile': modelfile,
        'train_data': trainf,
        'test_data': testf,
        'valid_data': validatef,
        'padding_size': 128
    }
else:
    modelfile = "model/qtSimSharedTwinMidd_{epoch:02d}_{mae:.4f}.h5"
    config = {
        'maxlen': 128,
        'mf': "model/qt_sim_model.h5",
        'batch_size': 500,  # 256 批次大小
        'valid_length': 100000,  # 验证集总大小
        'epochs': 100,
        'use_gpu': use_gpu,
        'gpus': gpus,
        'pretrained_path': 'data/chinese_simbert_L-6_H-384_A-12',  # chinese_simbert_L-12_H-768_A-12',
        'train': True,
        'modelfile': modelfile,
        'train_data': trainf,
        'test_data': testf,
        'valid_data': validatef,
        'padding_size': 128
    }

maxlen = config['maxlen']
valid_length = config['valid_length']
padding_size = config['padding_size']
pretrained_path = config.get('pretrained_path')
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
dict_path = os.path.join(pretrained_path, 'vocab.txt')

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
            # 归一化label
            if label == 1:
                label = 1
            elif label == 2:
                label = 1
            D.append((query, title, label))
            #D.append((title, query, label))
    return D


# 加载数据集
train_data = config['train_data']
valid_data = config['valid_data']
test_data = config['test_data']

train_data = load_data(train_data)
valid_data = load_data(valid_data)
test_data = load_data(test_data)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        title_batch_token_ids, title_batch_segment_ids = [], []
        for is_end, (query, title, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(query, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            token_ids, segment_ids = tokenizer.encode(title, maxlen=maxlen)
            title_batch_token_ids.append(token_ids)
            title_batch_segment_ids.append(segment_ids)

            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                title_batch_token_ids = sequence_padding(title_batch_token_ids)
                title_batch_segment_ids = sequence_padding(title_batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids, title_batch_token_ids, title_batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
                title_batch_token_ids, title_batch_segment_ids = [], []

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,  # with_nsp=False, with_mlm=True,
    model='bert',
    application='encoder',
    with_mlm=True,
    return_keras_model=False,
)

# 转换数据集
batch_size = config['batch_size']
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

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.weights')
            model.save(config['mf'])
            print("-->Save Model Successfully!" + config['mf'])
        test_acc = evaluate(test_generator)
        print('-->val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
              (val_acc, self.best_val_acc, test_acc)
              )

if __name__ == '__main__':
    print('--->python {} \nconfig:{}\n\n'.format(sys.argv[0], config))
    train = config['train']
    modelfile = config['modelfile']
    epochs = config['epochs']
    gpus = config['gpus']

    tfback._get_available_gpus = _get_available_gpus
    devices = []
    if config['use_gpu'].strip() != '':
        arr = config['use_gpu'].strip().split(',')
        for i in arr:
            devices.append('/gpu:{}'.format(i))
    print('-->gpu:', devices)
    # 自适应分配显存
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    KTF.set_session(session)

    # evaluator = Evaluator()
    tx_in = Input(shape=(None,), name='Input-Token-title')
    ts_in = Input(shape=(None,), name='Input-Segment-title')
    qx_in = Input(shape=(None,), name='Input-Token-query')
    qs_in = Input(shape=(None,), name='Input-Segment-query')

    title_model = Model(inputs=bert.model.input, outputs=bert.model.output[0])
    query_model = Model(inputs=bert.model.input, outputs=bert.model.output[0])
    title_vec = title_model([tx_in, ts_in])
    query_vec = query_model([qx_in, qs_in])
    print('query_vec:',query_vec)
    print('title_vec:',title_vec)
    print('type bert.model:', type(bert.model))
    print('type query_vec:', type(query_vec))
    checkpoint = ModelCheckpoint(modelfile, monitor='mae', mode='min', verbose=1, period=1, save_best_only=True)
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    with strategy.scope():
        outputs = Lambda(cosine_distance, name='title-query-cosin')([query_vec, title_vec])
        print('-->distance:', outputs)
        model = keras.models.Model(inputs=[tx_in, ts_in, qx_in, qs_in], outputs=outputs)

        if gpus > 0:
            model = multi_gpu_model(model, gpus=gpus)

        model.summary()
        for i, layer in enumerate(model.layers):
            print(i, layer, layer.name, layer.trainable)
            if i > 10:
                break
        model.compile(
            loss='mse', optimizer=Adam(2e-5),  # 用足够小的学习率
            # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
            metrics=['mae'],
        )

    if train:
        model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            verbose=1,
            callbacks=[checkpoint],
            validation_data=valid_generator.forfit(), validation_steps=int(valid_length / batch_size)
        )
        print(u'--->final test acc: %05f\n' % (evaluate(test_generator)))
