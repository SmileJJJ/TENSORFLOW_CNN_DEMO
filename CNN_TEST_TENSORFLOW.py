#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-
"""
__title__ = ’操作TensorFlow计算图'
            '实现卷积层，池化层，激活函数，全连接层等操作'
__author__ = 'NANXI'
__mtime__ = '2018.12.10'
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
import sys


'''
输入数据
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
卷积层
'''
kernel_width = 3
kernel_height = 3
kernel_num = 1
#卷积核的状态
#卷积核宽，卷积核高，输入图片的颜色通道，输出图片的数量(卷积核数量)
kernel_shape = [kernel_width,kernel_height,input_channel,kernel_num]
# 随机创建卷积核
#tf.Variable(初始化参数，名称)
#tf.truncated_normal(shape=张量维度,mean=均值,stddev=标准差)
# kernel = tf.Variable(tf.truncated_normal(shape=shape,mean=0,stddev=0.01))
kernel_val = tf.constant([[1,1,1],[1,2,1],[1,1,1]],dtype=tf.float32)
kernel = tf.reshape(kernel_val,shape=kernel_shape,name='kernel')

#TensorFlow提供的卷积操作
#tf.nn.conv2d
#input:输入的要做卷积的图片，要求为一个张量，shape为[图片数量，图片高度，图片宽度，图片通道数]
#filter:卷积核，要求为一个张量，shape为[卷积核高度，卷积核宽度，图像通道数，卷积核数量]
#strides:卷积在图像每一维的补偿，这是一个一维向量，[1，stride，stride，1]
#padding:卷积方式,'SAME'OR'VALID'
#use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为True

cnn_layer = tf.nn.conv2d(input=input,filter=kernel,strides=[1,1,1,1],padding='SAME')
# cnn_layer = tf.reshape(cnn_layer,shape=[4,5])
# [[ 6.  9. 10. 13. 13.]
#  [ 9. 14. 15. 20. 13.]
#  [10. 15. 18. 13.  9.]
#  [ 7. 12.  9.  9.  4.]]

'''
池化层
'''
#TensorFlow提供的最大池化操作
#tf.nn.max_pool
#input:需要池化的数组矩阵，通常是卷积后的feature map，保持[batch,height,width,channels]的shape
#ksize:池化窗口大小，一般为[1,height,width,1],只在height和width上做池化操作
#strides:池化窗口的步长，一般为[1,strid,strid,1]
#padding:有'SAME'和'VALID'两种填充方式，前者表示行/列数不足则填充0，后者则直接舍弃不足的那几行/列
#use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为True
pool_layer = tf.nn.max_pool(cnn_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
pool_layer = tf.reshape(pool_layer,shape=[2,3])
pool_layer = tf.constant([[14,-20,13],[15.6,20,-5]])
# [[14. 20. 13.]
#  [15. 18.  9.]]

'''
激活层
'''
#TensorFlow提供的激活函数RELU
#将输入矩阵进行激活，小于0的数值变成0，大于0的保持不变
active_layer = tf.nn.relu(pool_layer)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    cnn_result = sess.run(active_layer)
    print(type(cnn_result))
    print(cnn_result.shape)
    print(cnn_result)




