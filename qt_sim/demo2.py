# encoding=utf8
import sys

import jieba

#jieba.add_word('740')
#jieba.add_word('740i')

res = jieba.cut('全新宝马740i，面不改色，豪华升级')
print('|'.join(res))