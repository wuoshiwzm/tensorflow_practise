import numpy as np
import tensorflow as tf
# 分类网络类
class TextCNN(object):
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,filter_sizes,num_filters,l2_reg_lambda):
        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name='input_x')  #N*128
        self.input_y = tf.placeholder(tf.float32,[None,num_classes],name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

        # L2 loss
        l2_loss = tf.constant(0.0)

        # embedding layer 生成device实例
        with tf.device('cpu:0'),tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W,self.input_x)

