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

set_gelu('tanh')  # 切换gelu版本
savedir = 'model'
datadir = 'data'
filepath = "qtSim_cosin_bert_mlp_middle_{epoch:02d}_{mae:.4f}.h5"
modelfile = os.path.join(savedir, filepath)
trainf, testf, validatef = datadir + '/train_data.txt', datadir + '/test_data.txt', datadir + '/validate_data.txt'

isUsegpu = False
if isUsegpu:
    use_gpu = '0,1'
    gpus = 2
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpus = 0
    use_gpu = ''

config = {
    'maxlen': 128,
    'mf': "model/qt_sim_model.h5",
    'batch_size': 256,  # 批次大小
    'valid_size': 100000, # 验证集总大小
    'epochs': 30,
    'use_gpu': use_gpu,
    'gpus': gpus,
    'pretrained_path': 'models/chinese_simbert_L-6_H-384_A-12', # chinese_simbert_L-12_H-768_A-12',
    'train': True,
    'modelfile': modelfile,
    'train_data': trainf,
    'test_data': testf,
    'valid_data': validatef,
    'padding_size': 128
}

maxlen = config['maxlen']
valid_size = config['valid_size']
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
            D.append((title, query, label))
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
        for is_end, (query, title, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                query, title, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids, length=padding_size)
                batch_segment_ids = sequence_padding(batch_segment_ids, length=padding_size)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids, batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,  # with_nsp=False, with_mlm=True,
    model='bert',
    application='encoder',
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

if __name__ == '__main__1':
    gen = train_generator.forfit()
    for i in range(1000):
        arr, b = next(gen)
        print(len(arr[1]))
        print(len(arr[0]))

if __name__ == '__main__':
    print('--->python {} \nconfig:{}\n'.format(sys.argv[0], config))
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
    # ----------- MLP ---------------------------------------------------------------------------------
    input_len = padding_size
    x_in = Input(shape=(input_len,), name='Input-Token-MLP')
    s_in = Input(shape=(input_len,), name='Input-Segment-MLP')
    concated_vec = Concatenate(axis=1, name='Token-Segment-Concate')([x_in, s_in])
    # emb = Embedding(input_dim=23000, output_dim=256, name='embedding')(concated_vec)

    qt_vec = Dense(units=768, name='qt-dense-1')(concated_vec)
    qt_vec = BatchNormalization(name='norm-1')(qt_vec)
    qt_vec = Activation('relu')(qt_vec)

    qt_vec = Dense(units=768, name='qt-dense-2')(qt_vec)
    qt_vec = BatchNormalization(name='norm-2')(qt_vec)
    qt_vec = Activation('relu')(qt_vec)

    qt_vec = Dense(units=512, activation='relu', name='qt-dense-3')(qt_vec)
    qt_vec = BatchNormalization(name='norm-3')(qt_vec)
    qt_vec = Activation('relu')(qt_vec)

    qt_vec = Dense(units=128, activation='relu', name='qt-dense-4')(qt_vec) # none,128
    #qt_vec = Reshape([-1], name='qt-reshape')(qt_vec)

    # ----------- bert model ---------------------------------------------------------------------------
    query_output = Dropout(rate=0.5, name='query-drop-1')(bert.model.output)
    title_output = Dropout(rate=0.5, name='title-drop-1')(bert.model.output)
    query_output = Dense(units=384, activation='relu', kernel_initializer=bert.initializer, name='query-dense-1')(
        query_output)
    title_output = Dense(units=384, activation='relu', kernel_initializer=bert.initializer, name='title-dense-1')(
        title_output)

    query_output = Dropout(rate=0.5, name='query-drop-2')(query_output)
    title_output = Dropout(rate=0.5, name='title-drop-2')(title_output)
    query_output = Dense(units=384, activation='relu', kernel_initializer=bert.initializer, name='query-dense-2')(
        query_output)
    title_output = Dense(units=384, activation='relu', kernel_initializer=bert.initializer, name='title-dense-2')(
        title_output)

    query_output = Dropout(rate=0.5, name='query-drop-3')(query_output)
    title_output = Dropout(rate=0.5, name='title-drop-3')(title_output)
    query_output = Dense(units=256, activation='relu', kernel_initializer=bert.initializer, name='query-dense-3')(
        query_output)
    title_output = Dense(units=256, activation='relu', kernel_initializer=bert.initializer, name='title-dense-3')(
        title_output)

    checkpoint = ModelCheckpoint(modelfile, monitor='mae', mode='min', verbose=1, period=1)  # save_best_only=True
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    with strategy.scope():
        query_output = Concatenate(axis=1, name='query-mlp')([query_output, qt_vec])
        title_output = Concatenate(axis=1, name='title-mlp')([title_output, qt_vec])

        output = Lambda(cosine_distance, name='title-query-cosin')([query_output, title_output])
        print('-->distance:', output)
        print('-->bert.model.input type:', type(bert.model.input), ',value:',bert.model.input)
        inpus = bert.model.input
        inpus.append(x_in)
        inpus.append(s_in)
        print('-->qt sim input type:', type(bert.model.input), ',value:', bert.model.input)
        model = keras.models.Model(inputs=inpus, outputs=output)

        if gpus > 0:
            model = multi_gpu_model(model, gpus=gpus)

        model.summary()
        for i, layer in enumerate(model.layers):
            print(i, layer, layer.name, layer.trainable)
            if i > 10:
                break
        model.compile(
            loss='mse',
            optimizer=Adam(2e-5),  # 用足够小的学习率
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
            validation_data=valid_generator.forfit(), validation_steps=int(valid_size / batch_size)
        )
        print(u'--->final test acc: %05f\n' % (evaluate(test_generator)))
