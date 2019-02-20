# 训练主函数
print('etst')
import tensorflow as tf
import numpy as  np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Params  参数
tf.flags.DEFINE_float('dev_sample_percetage', 1, 'percentage training data in validation loop')
# 正/负例数据
tf.flags.DEFINE_string('positive_data_fill', './data/rt-polaritydata/rt-polarity.pos', 'positive samples')
tf.flags.DEFINE_string('negative_data_fill', './data/rt-polaritydata/rt-polarity.neg', 'negative samples')
# hyperparams
# 中间隐层的维度 每一个单词转换为对应的维度  默认为128
tf.flags.DEFINE_integer('embedding', 128, 'Dimension for character')
# 多个单词一卷积 一个单词对应input的一行
tf.flags.DEFINE_string('filter_size', '3,4,5', 'filter size')
tf.flags.DEFINE_integer('num_filters', 128, 'how many Filters here')
tf.flags.DEFINE_float('drop_out', 0.5, 'Dropout')
tf.flags.DEFINE_float('L2_reg_lambda', 0.0, 'L2')  # 正则化

# 训练参数
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_integer('num_epochs', 200, 'how many total epochs')
tf.flags.DEFINE_integer('evaluate_every', 100, 'evaluate every time')  # 多少次迭代做一个可视化显示
tf.flags.DEFINE_integer('checkpoint_every', 100, 'save model every ...')  # 每个多少次保留一次模型

# # Flags生效
FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags() 弃用了
FLAGS.flag_values_dict()
print('\nParameters:')
for attr, value in sorted(FLAGS._flags().items()):
    print(attr.upper(),'=',value)
    # print('{}={}').format(attr.upper(), value)

x_text,y = data_helpers.load_data_and_labels(FLAGS.positive_data_fill,FLAGS.negative_data_fill)