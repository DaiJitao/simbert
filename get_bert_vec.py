# ！-*- coding: utf-8 -*-
# SimBERT 相似度任务测试
# 基于LCQMC语料

import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import uniout, open
import logging
from keras.models import Model
import platform

os_p = platform.system().lower()
if os_p == 'linux':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf8')
    print('===============> platform {}'.format(os_p))
elif os_p == 'windows':
    print('===============> platform {}'.format(os_p))



import os  # djt

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # djt

maxlen = 32

is_simbert = True
if is_simbert:
    # bert配置
    config_path = 'D:/qichezhijia/workspace/pycharm/tensorflow_worksapce/tfVersion_1.15.2/simbert/models/chinese_simbert_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'D:/qichezhijia/workspace/pycharm/tensorflow_worksapce/tfVersion_1.15.2/simbert/models/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
    dict_path ='D:/qichezhijia/workspace/pycharm/tensorflow_worksapce/tfVersion_1.15.2/simbert/models/chinese_simbert_L-12_H-768_A-12/vocab.txt'

else:
    # 原生bert chinese_L-12_H-768_A-12
    config_path = './models/chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'models/chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = './models/chinese_L-12_H-768_A-12/vocab.txt'

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

print('--->outputs:', bert.model.outputs)
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


def get_vec_by_query_pair(text, second_text):
    """提取文本向量
    """
    if second_text == None or second_text.strip() == '':
        raise Exception('second_text is empty!')

    token_ids, segment_ids = tokenizer.encode(text, second_text, max_length=maxlen)
    #print("token_ids={},segment_ids={}".format(token_ids, segment_ids))
    vec = encoder.predict([[token_ids], [segment_ids]])[0] # 获取bert向量
    return list(vec)

def get_vec_by_query(text):
    """提取文本向量
    """
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    #print("token_ids={},segment_ids={}".format(token_ids, segment_ids))
    vec = encoder.predict([[token_ids], [segment_ids]])[0] # 获取bert向量
    return list(vec)


def save_query(infile, outfile):
    '''
    获取向量
    :return:
    '''
    outfile_fp = open(outfile, encoding='utf-8', mode='w')
    with open(infile, 'r', encoding='utf-8') as fp:
        # lines = fp.readlines()
        for line in fp:
            text = line.strip()
            if text == '' or len(text) == 0:
                continue
            vec = get_vec_by_query(text)
            outfile_fp.write("{}\t{}\n".format(text, vec))
    outfile_fp.close()
    print('=============>写入文件：{}'.format(outfile))


def save_query2(infile, outfile):
    '''
    获取向量
    :return:
    '''
    logging.info("============================>open {}".format(outfile))
    outfile_fp = open(outfile, encoding='utf-8', mode='w')
    with open(infile, 'r', encoding='utf-8') as fp:
        print('====================>read infile {} ok!'.format(infile))
        count = 0
        for line in fp:
            count += 1
            try:
                temp = line.strip().split('\t')
                query = temp[1].strip()  # 获取query
                if query != '' :
                    vec = get_vec_by_query(query)
                    id = temp[0].strip()
                    outfile_fp.write("{}\t{}\n".format(id, vec))
                if count % 100 == 0:
                    logging.info("======================>process line:{}, query:{}".format(count, line.strip()))
            except Exception as e:
                print(e)

    outfile_fp.close()
    print('=============>写入文件：{}'.format(outfile))

if __name__ == '__main__':
    text = '我爱你'
    print(get_vec_by_query(text))
    print(get_vec_by_query_pair(text, second_text='----------------'))
    print(get_vec_by_query_pair(text, second_text='我和你是不是还不好呢'))

if __name__ == '__main__1':
    infile = 'data/query_data/noclcik_query0728.clean.txt'
    outfile = 'data/query_data/noclick.query0728.vec'
    save_query(infile, outfile)

if __name__ == '__main__2':
    """sample代码"""
    name = 'noclcik_query0728.vec'
    infile = 'query_data/noclcik_query0728.clean.txt'
    outfile = 'query_data/' + name
    query = '月 销量 最好'
    v1 = get_vec_by_query(query)
    print('============>无点击query：{} '.format(query))
    print(v1)
    save_query(infile, outfile)
    # fp = open(outfile, 'r',encoding='utf-8')
    # for i in fp:
    #     print(i.split("\t")[-1])
