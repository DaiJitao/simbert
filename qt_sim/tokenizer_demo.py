# encoding=utf8
import sys
import os

from bert4keras.tokenizers import Tokenizer

simBert = True
if simBert:
    pretrained_path = '../models/chinese_simbert_L-12_H-768_A-12'  # models/chinese_L-12_H-768_A-12
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    dict_path = os.path.join(pretrained_path, 'vocab.txt')
else:
    pretrained_path = '../models/chinese_L-12_H-768_A-12'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    dict_path = os.path.join(pretrained_path, 'vocab.txt')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def demoPrint(query, token_ids, segment_ids):
    print('query_length:', len(query))
    print(token_ids, segment_ids)
    print('token_ids_length:', len(token_ids))


query = '2022款宝马X5值不值得买?上了不限速高速后一目了'
token_ids, segment_ids = tokenizer.encode(query, max_length=128)
tokens = tokenizer.tokenize(query)
print(tokens)
demoPrint(query, token_ids, segment_ids)
print()
query = '宝骏730自动挡新款六座'
token_ids, segment_ids = tokenizer.encode(query, max_length=128)
tokens = tokenizer.tokenize(query)
print(tokens)
demoPrint(query, token_ids, segment_ids)