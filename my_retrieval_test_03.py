# ！-*- coding: utf-8 -*-
# SimBERT 相似度任务测试
# 基于LCQMC语料

import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import uniout, open
from keras.models import Model

maxlen = 32

# bert配置
config_path = './models/chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'models/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './models/chinese_simbert_L-12_H-768_A-12/vocab.txt'

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
    """
    text1,text2,label
    :param filename:
    :return:
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D


# 加载数据集
# train_data = load_data('datasets/lcqmc/lcqmc.train.data')
valid_data = load_data('./data/lcqmc_data/test.txt')  # text1,text2,label


# test_data = load_data('datasets/lcqmc/lcqmc.test.data')

# 测试相似度效果
# data = valid_data
# a_token_ids, b_token_ids, labels = [], [], []
# texts = []

# for d in data: # (text1,text2,label)
#     token_ids = tokenizer.encode(d[0], max_length=maxlen)[0]
#     a_token_ids.append(token_ids)
#     token_ids = tokenizer.encode(d[1], max_length=maxlen)[0]
#     b_token_ids.append(token_ids)
#     labels.append(d[2])
#     texts.extend(d[:2])
#
# a_token_ids = sequence_padding(a_token_ids)
# b_token_ids = sequence_padding(b_token_ids)
# a_vecs = encoder.predict([a_token_ids, np.zeros_like(a_token_ids)],
#                          verbose=True)
# b_vecs = encoder.predict([b_token_ids, np.zeros_like(b_token_ids)],
#                          verbose=True)
# labels = np.array(labels)

# a_vecs = a_vecs / (a_vecs**2).sum(axis=1, keepdims=True)**0.5
# b_vecs = b_vecs / (b_vecs**2).sum(axis=1, keepdims=True)**0.5
# sims = (a_vecs * b_vecs).sum(axis=1)
#
# # 以0.9为阈值，acc为79.82%
# print('acc:', ((sims > 0.9) == labels.astype('bool')).mean())

# 测试全量检索能力
# vecs = np.concatenate([a_vecs, b_vecs], axis=1).reshape(-1, 768)


def most_similar(text, topn=10):
    """检索最相近的topn个句子
    """
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    # print("token_ids={},segment_ids={}".format(token_ids, segment_ids))

    vec = encoder.predict([[token_ids], [segment_ids]])[0]
    print(vec)
    # print('vec size={}, element={}'.format(len(vec), vec[-1]))
    # print('===========>之前vec={}'.format(vec[:5]))
    # vec /= (vec**2).sum()**0.5
    # print('===========>之后vec={}'.format(vec[:5]))
    try:
        print('vec size={}, element={}'.format(len(vec), vec[-1]))
    except Exception as e:
        print(e)
        # sims = np.dot(vecs, vec)
        # return [(texts[i], sims[i]) for i in sims.argsort()[::-1][:topn]]


def get_vec_by_query(text):
    """提取文本向量
    """
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    print(text)
    print(token_ids, segment_ids)
    vec = encoder.predict([[token_ids], [segment_ids]])[0]
    return list(vec)



def save_query(infile, outfile):
    '''
    数据编码处理
    :return:
    '''
    outfile_fp = open(outfile, encoding='utf-8', mode='w')
    with open(infile,'r',encoding='utf-8') as fp:
        # lines = fp.readlines()
        for line in fp:
            text = line.strip()
            if text == '' or len(text) == 0:
                continue
            text = A + B
            vec = get_vec_by_query(text)
            outfile_fp.write("{}\t{}".format(text, vec))
            outfile_fp.write("\n")
    outfile_fp.close()
    print('写入文件：{}'.format(outfile))


if __name__ == '__main__':
    name = 'click.data.03'
    infile = 'C:/Users/daijitao/Desktop/upload2/data/click.data/' + name
    outfile = 'data/query_data/out.' + name + '.txt'
    save_query(infile, outfile)
    # fp = open(outfile, 'r',encoding='utf-8')
    # for i in fp:
    #     print(i.split("\t")[-1])

"""
>>> most_similar(u'怎么开初婚未育证明', 20)
[
    (u'开初婚未育证明怎么弄？', 0.9728098),
    (u'初婚未育情况证明怎么开？', 0.9612292),
    (u'到哪里开初婚未育证明？', 0.94987774),
    (u'初婚未育证明在哪里开？', 0.9476072),
    (u'男方也要开初婚证明吗?', 0.7712214),
    (u'初婚证明除了村里开，单位可以开吗？', 0.63224965),
    (u'生孩子怎么发', 0.40672967),
    (u'是需要您到当地公安局开具变更证明的', 0.39978087),
    (u'淘宝开店认证未通过怎么办', 0.39477515),
    (u'您好，是需要当地公安局开具的变更证明的', 0.39288986),
    (u'没有工作证明，怎么办信用卡', 0.37745982),
    (u'未成年小孩还没办身份证怎么买高铁车票', 0.36504325),
    (u'烟草证不给办，应该怎么办呢？', 0.35596085),
    (u'怎么生孩子', 0.3493368),
    (u'怎么开福利彩票站', 0.34158638),
    (u'沈阳烟草证怎么办？好办不？', 0.33718678),
    (u'男性不孕不育有哪些特征', 0.33530876),
    (u'结婚证丢了一本怎么办离婚', 0.33166665),
    (u'怎样到地税局开发票？', 0.33079252),
    (u'男性不孕不育检查要注意什么？', 0.3274408)
]
"""
