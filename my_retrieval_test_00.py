# ！-*- coding: utf-8 -*-
# SimBERT 相似度任务测试
# 基于LCQMC语料

import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
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


def get_vec_by_query(text):
    """提取文本向量
    """
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    # print("token_ids={},segment_ids={}".format(token_ids, segment_ids))
    vec = encoder.predict([[token_ids], [segment_ids]])[0]
    print('vec size={}, element={}'.format(len(vec), vec[-1]))
    print(type(vec))
    return list(vec)

        # print(vec)
        # print('vec size={}, element={}'.format(len(vec), vec[-1]))
        # print('===========>之前vec={}'.format(vec[:5]))
        # vec /= (vec**2).sum()**0.5
        # print('===========>之后vec={}'.format(vec[:5]))

        # sims = np.dot(vecs, vec)
        # return [(texts[i], sims[i]) for i in sims.argsort()[::-1][:topn]]


def save_query(infile, outfile):
    '''
    获取向量
    :return:
    '''
    outfile_fp = open(outfile, encoding='utf-8', mode='w')
    with open(infile,'r',encoding='utf-8') as fp:
        # lines = fp.readlines()
        for line in fp:
            text = line.strip()
            if text == '' or len(text) == 0:
                continue
            vec = get_vec_by_query(text)
            outfile_fp.write("{}\t{}\n".format(text, vec))
    outfile_fp.close()
    print('=============>写入文件：{}'.format(outfile))


if __name__ == '__main__':
    """sample代码"""
    name = 'sample.noclick.query.vec2'
    infile = 'data/query_data/noclick_query.cleaned.data2'
    outfile = 'data/query_data/' + name
    query = '月 销量 最好'
    v1 = get_vec_by_query(query)
    print('============>无点击query：{} '.format(query))
    print(v1)
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
