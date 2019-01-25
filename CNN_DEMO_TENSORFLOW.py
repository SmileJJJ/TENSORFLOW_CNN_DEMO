#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-
"""
__title__ = ’卷积神经网络'
__author__ = 'NANXI'
__mtime__ = '2018.12.17'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import numpy as np
import tensorflow as tf

class Cnn_filter(object):

    def __init__(self,sign,width,height,input_channel,count):
        self.num = sign
        self.size_heigh = height
        self.size_width = width
        self.num = count
        self.kernel_shape = [width,height,input_channel,count]

class Cnn_conv_layer(object):

    def __init__(self,input,cnn_filter,
                 pool_size,pool_strides,
                 kernel_padding='SAME',
                 pool_padding='SAME',
                 pool_strategy='MAX'):
        self.input = input

        kernel_val = tf.constant([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=tf.float32)
        self.cnn_filter = tf.reshape(kernel_val,shape=cnn_filter.kernel_shape,name='kernel')
        # self.cnn_filter =tf.Variable(tf.truncated_normal(shape=cnn_filter.kernel_shape,
        #                                                  mean=0,
        #                                                  stddev=0.01))
        self.padding = kernel_padding
        self.pool_strategy = pool_strategy
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.pool_padding = pool_padding

    def create_cnn_layer(self):
        self.cnn_layer = tf.nn.conv2d(input = self.input,
                                 filter=self.cnn_filter,
                                 strides=[1,1,1,1],
                                 padding=self.padding)
        return self.cnn_layer

    def create_pool_layer(self):
        if self.pool_strategy == 'MAX':
            self.pool_layer = tf.nn.max_pool(self.cnn_layer,
                                        ksize=[1,self.pool_size[0],self.pool_size[1],1],
                                        strides=[1,self.pool_strides[0],self.pool_strides[1],1],
                                        padding='SAME')
        return self.pool_layer

'''
解析输入数据
'''
input_height = 4
input_width = 5
input_channel = 1
input_num = 1
#TensorFlow里要求输入数据的shape为[图片数量，图片高度，图片宽度，图片通道]
input_shape = [input_num,input_height,input_width,input_channel]
input_data = tf.constant([[0,1,1,2,4],[3,2,1,2,1],[0,0,4,1,2],[2,3,0,1,0]],dtype=tf.float32)
input = tf.reshape(input_data,shape=input_shape,name='input')

'''
配置神经网络
'''
#卷积层1
#卷积核(标号,宽,高,输入通道,数量)
cnn_filter_1 = Cnn_filter(1,3,3,input_channel,1)
#卷积层2
# cnn_filter_2 = Cnn_filter(2,5,5,36)

'''
配置TensorFlow计算图
'''
#卷积层1
conv_layer_1 = Cnn_conv_layer(input,cnn_filter_1,
                              pool_size=[2,2],
                              pool_strides=[2,2])
cnn_layer = conv_layer_1.create_cnn_layer()
pool_layer = conv_layer_1.create_pool_layer()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(pool_layer))
