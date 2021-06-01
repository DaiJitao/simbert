# encoding=utf8
import sys

import tensorflow as tf
from keras.layers import Dense, Embedding, AveragePooling2D, AveragePooling1D, Input
from keras.models import Model, Sequential


x=tf.Variable(tf.random.normal([2,9,1]))
avg = AveragePooling1D(pool_size=2, strides=1, padding='valid')(x)
print(avg)