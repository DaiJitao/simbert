#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l

from __future__ import print_function
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pdb
import os
import tensorflow as tf
import keras
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#os.environ['TF_KERAS'] = '1'

import horovod.keras as hvd
# Initialize Horovod
hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

# 基本参数
maxlen = 512
batch_size = 24
epochs = 20

# bert配置
model_path = "./chinese_roberta_wwm_ext_L-12_H-768_A-12"
config_path = model_path + '/bert_config.json'
checkpoint_path = model_path + '/bert_model.ckpt'
dict_path = model_path + '/vocab.txt'

#config_path = '/dmcv1/zhouhui/pretrained_model/tf_unilm_model/config.json'
#checkpoint_path = '/dmcv1/zhouhui/pretrained_model/tf_unilm_model/bert_model.ckpt'
#dict_path = '/dmcv1/zhouhui/pretrained_model/tf_unilm_model/vocab.txt'

model_save_path = './output/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            object_uid, title, content = l.strip().split('\t')
            D.append((title, content))
    return D


# 加载数据集
train_data = load_data('./dataset/train_with_series_and_replace_space.txt')
valid_data = load_data('./dataset/test_with_series_and_replace_space.txt')
test_data = load_data('./dataset/test_with_series_and_replace_space.txt')

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


#def save_vocab(dict_path, token_dict, encoding='utf-8'):
#    """将词典（比如精简过的）保存为文件
#    """
#    with open(dict_path, 'w', encoding=encoding) as writer:
#        for k, v in sorted(token_dict.items(), key=lambda s: s[1]):
#            writer.write(k+'\t'+str(v) + '\n')
#
#save_vocab('token_dict.txt', token_dict)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (title, content) in self.sample(random):
            #pdb.set_trace()
            #content = "中国[]"
            token_ids, segment_ids = tokenizer.encode(
                content, title, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss
#strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
#print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
#with strategy.scope():
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
#opt = keras.optimizers.Adadelta(1e-5 * hvd.size())
opt = Adam(1e-5 * hvd.size())
# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)
model.compile(optimizer=opt,
              #experimental_run_tf_function=False
              )
model.summary()


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= 14:
            metrics = self.evaluate(valid_data)  # 评测模型
            if metrics['bleu'] > self.best_bleu:
                self.best_bleu = metrics['bleu']
                model.save_weights(os.path.join(model_save_path, 'best_model.weights'))  # 保存模型
            metrics['best_bleu'] = self.best_bleu
            print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for title, content in tqdm(data):
            total += 1
            title = ' '.join(title)
            pred_title = ' '.join(autotitle.generate(content, topk))
            if pred_title.strip():
                try:
                    scores = self.rouge.get_scores(hyps=pred_title, refs=title)
                except:
                    print(title,content)
                    break
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[title.split(' ')],
                    hypothesis=pred_title.split(' '),
                    smoothing_function=self.smooth
                )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
    if hvd.rank() == 0:
        callbacks.append(evaluator)
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) // hvd.size(),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1 if hvd.rank() == 0 else 0
    )
    
    #model.load_weights(os.path.join(model_save_path, 'best_model.weights'))
    print('test_data:', evaluator.evaluate(test_data))

else:

    model.load_weights('./best_model.weights')
