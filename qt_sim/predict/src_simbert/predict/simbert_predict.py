#! -*- coding: utf-8 -*-
# SimBERT训练代码
# 训练环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7

from __future__ import print_function
import json
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import text_segmentate
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout
import keras.backend.tensorflow_backend as tfback
from tensorflow.python.keras import backend as KTF
from keras.utils import multi_gpu_model
from keras.models import load_model
import os
import sys

isTestPlatform = True
if isTestPlatform:
    use_gpu = ''
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu
    gpus = 0
else:
    use_gpu = '0,1,2,3'
    gpus = 4

config = {}
# 基本信息
maxlen = 128

# bert配置
bert_dir = 'data/'
config_path = bert_dir + 'chinese_simbert_L-12_H-768_A-12/bert_config.json'  # models/chinese_simbert_L-12_H-768_A-12/bert_config.json
checkpoint_path = bert_dir + 'chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = bert_dir + 'chinese_simbert_L-12_H-768_A-12/vocab.txt'
config['config_path'] = config_path
config['checkpoint_path'] = checkpoint_path
config['dict_path'] = dict_path
config['maxlen'], config['gpus'] = maxlen, gpus

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

def read_corpus(corpus_path):
    """读取语料，每行一个json
    """
    while True:
        with open(corpus_path) as f:
            for l in f:
                data = l.strip()
                if data != '':
                    try:
                        res = {}
                        arr = data.split('\t')
                        text = arr[0]
                        synonyms = arr[1:]
                        res['text'],  res['synonyms'] = text, synonyms
                        yield res
                    except Exception as e:
                        print(e)
                        print('data error:', data)


def truncate(text):
    """截断句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen - 2, seps, strips)[0]


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.some_samples = []

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, d in self.sample(random):
            text, synonyms = d['text'], d['synonyms']
            synonyms = [text] + synonyms
            np.random.shuffle(synonyms)
            text, synonym = synonyms[:2]
            text, synonym = truncate(text), truncate(synonym)
            self.some_samples.append(text)
            if len(self.some_samples) > 1000:
                self.some_samples.pop(0)
            token_ids, segment_ids = tokenizer.encode(
                text, synonym, max_length=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            token_ids, segment_ids = tokenizer.encode(
                synonym, text, max_length=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class TotalLoss(Loss):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """
    def compute_loss(self, inputs, mask=None):
        loss1 = self.compute_loss_of_seq2seq(inputs, mask)
        loss2 = self.compute_loss_of_similarity(inputs, mask)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='similarity_loss')
        return loss1 + loss2

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, y_pred, _ = inputs
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
        similarities = similarities * 30  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = K.equal(idxs_1, idxs_2)
        labels = K.cast(labels, K.floatx())
        return labels

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    return_keras_model=False,
)

class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n,
                                        topk)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)

def gen_synonyms(text, n=100, k=20, model=None):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    效果：
        >>> gen_synonyms(u'微信和支付宝哪个好？')
        [
            u'微信和支付宝，哪个好?',
            u'微信和支付宝哪个好',
        ]
    """
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    if model != None:
        myp = model.predict([X, S])
        print("myp:",myp)
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]

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

def just_show():
    """随机观察一些样本的效果
    """
    some_samples = train_generator.some_samples
    S = [np.random.choice(some_samples) for i in range(3)]
    for s in S:
        try:
            print(u'原句子：%s' % s)
            print(u'同义句子：')
            print(gen_synonyms(s, 10, 10))
            print()
        except:
            pass

class Evaluate(keras.callbacks.Callback):
    """评估模型
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        #model.save_weights('qt_sim_models/latest_model.weights')
        #mf = modelfile.format(epoch, '%.5f' %logs['loss'])
        #model.save(mf)
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            #model.save_weights('qt_sim_models/latest_model.weights')
            mf = modelfile.format(epoch,'%.5f' %logs['loss'])
            model.save(mf)
            print('-->saved model:{}'.format(mf))

        # 演示效果
        just_show()

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

def convert_data(text):
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen * 2)
    token_list = sequence_padding([token_ids])
    segment_list = sequence_padding([segment_ids])
    return [token_list, segment_list]

train = False
predict = True
batch_size = 64
all_data_size = 1015725
valid_size = 5
steps_per_epoch = int(all_data_size / batch_size) # 1000
epochs = 10
train_data = 'data/similar_qtt_from_train_data.txt'
validate_data = 'data/validate_data.txt'
train_generator = data_generator(read_corpus(train_data), batch_size)
valid_generator = data_generator(read_corpus(validate_data), batch_size)

modelfile = 'data/qtSimbertModelGPU_1_2.62564.h5' # 保存模型
config['modelfile'] , config['epochs'], config['train_data'] = modelfile, epochs, train_data
config['batch_size'], config['valid_size'] = batch_size,valid_size

if __name__ == '__main__':
    tfback._get_available_gpus = _get_available_gpus
    devices = []
    for i in use_gpu.split(','):
        if i != '':
            devices.append('/gpu:{}'.format(i))
    config['gpu_device'] = devices
    print('-->config:{}\n'.format(config))
    # 自适应分配显存
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    KTF.set_session(session)

    encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
    seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])

    evaluator = Evaluate()
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    with strategy.scope():
        print('bert.model.inputs={},bert.model.outputs={}'.format(bert.model.inputs, bert.model.outputs))
        outputs = TotalLoss([2, 3])(bert.model.inputs + bert.model.outputs)
        model = keras.models.Model(bert.model.inputs, outputs)
        #model.summary()

        if gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

        AdamW = extend_with_weight_decay(Adam, 'AdamW')
        optimizer = AdamW(learning_rate=2e-6, weight_decay_rate=0.01)
        model.compile(optimizer=optimizer)

    if train:
        print('train starting ...')
        model.fit_generator(
                train_generator.forfit(),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs, verbose=1,
                callbacks=[evaluator],
                # validation_data=valid_generator.forfit(), validation_steps=int(valid_size / batch_size)
            )
    else:
        modelfile = sys.argv[1] #'data/qtSimbertModelGPU_1_2.62564.h5'
        simbert_model = load_model(modelfile, {'TotalLoss': TotalLoss})
        simbert_model.summary()
        print()
        inf, outf = sys.argv[2], sys.argv[3]
        outfp = open(outf, mode='w', encoding='utf-8')
        print('\n--->inf: {}'.format(inf))
        with open(inf, 'r', encoding='utf-8') as fp:
            for line in fp:
                arr = line.strip().split('\t')
                if arr == 3:
                    text1, text2 = arr[0], arr[1]
                    data1, data2 = convert_data(text1), convert_data(text2)
                    vector1 = simbert_model.predict(data1)[0]
                    vector2 = simbert_model.predict(data2)[0]
                    cosin_score = cosine_distance(inputs=[tf.constant(vector1), tf.constant(vector2)])
                    dot_value = np.dot(vector2, np.transpose(vector1))
                    temp = '{}\t{}\t{}\n'.format(line.strip(), cosin_score, dot_value)
                    outfp.write(temp)
                    outfp.flush()

        outfp.close()
        print('ok saved in {}'.format(outf))

