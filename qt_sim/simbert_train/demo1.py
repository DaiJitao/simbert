# encoding=utf8
import sys
from keras.layers import Input
from bert4keras.snippets import sequence_padding
import numpy as np
from bert4keras.snippets import text_segmentate

maxlen = 128
def truncate(text):
    """截断句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen - 2, seps, strips)[0]

if __name__ == '__main__':
    text = '苹果怎么样,质量好不好啊？我想买！' * 13
    print(len(text))
    res = truncate(text)
    print(res)
    print(len(res))