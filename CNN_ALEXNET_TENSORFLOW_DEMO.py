#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-
"""
__title__ = 'None'
__author__ = 'None'
__mtime__ = 'None'
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

┌───┐   ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌
│Esc│   │ F1│ F2│ F3│ F4│ │ F5│ F6│ F7│ F8│ │ F9│F10│F11│F12│ │P/S│S L│P/B│  ┌┐    ┌┐    ┌┐  │
└───┘   └───┴───┴───┴───┘ └───┴───┴───┴───┘ └───┴───┴───┴───┘ └
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───────┐
│~ `│! 1│@ 2│# 3│$ 4│% 5│^ 6│& 7│* 8│( 9│) 0│_ -│+ =│ BacSp │ │Ins│Hom│PUp│ │N L│ / │ * │ - │   │
├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─────┤ 
│ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{ [│} ]│ | \ │ │Del│End│PDn│ │ 7 │ 8 │ 9 │   │   │
├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤ 
│ Caps │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  │                   │ 4  │ 5 │ 6 │   │   │
├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────────┤  
│ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│  Shift   │     │ ↑ │       │ 1  │ 2 │ 3 │   │   │
├─────┬──┴─┬─┴──┬┴───┴───┴───┴───┴───┴──┬┴───┼───┴┬────┬────┤ 
│ Ctrl│    │Alt │         Space         │ Alt│    │    │Ctrl│ │ ← │ ↓ │ → │ │   0   │ . │←─┘│    │
└─────┴────┴────┴───────────────────────┴────┴────┴────┴────┘ 
"""

import tensorflow as tf
import os

#TensorFlow里要求输入数据的shape为[图片数量，图片高度，图片宽度，图片通道]
image = tf.Variable(tf.random_normal([128, 227, 227, 3],
                                      dtype=tf.float32,
                                      stddev=1e-1))

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())

def Forward_propagation(image):
    '''
    前向传播
    :return: 返回第五层最大池化结果,每个卷积层的权重参数和偏置参数
    '''

    parameters = [] # 参数收集

    # 卷积层一：卷积运算
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(input=image,filter=kernel,strides=[1,4,4,1],padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
        # print_activations(conv1)

    # 卷积层一：LRN
    with tf.name_scope('lrn') as scope:
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # 卷积层一：最大池化层
    with tf.name_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(lrn1,
                              ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1],
                              padding='VALID',
                              name='pool1')
        # print_activations(pool1)
        #pool1/pool1   [128, 27, 27, 96]

    # 卷积层二：卷积运算
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=0.01),
                             name='weights')
        conv = tf.nn.conv2d(input=pool1, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),
                             trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name=scope)
        parameters += [kernel, biases]
        # print_activations(conv2)
        #conv2   [128, 27, 27, 256]

    # 卷积层二：LRN
    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # 卷积层二：最大池化层
    with tf.name_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(lrn2,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')
        # print_activations(pool2)
        # pool2/MaxPool   [128, 13, 13, 256]

    # 卷积层三：卷积运算
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],dtype=tf.float32,stddev=0.01),
                             name='weights')
        conv = tf.nn.conv2d(input=pool2, filter=kernel, strides= [1, 1, 1, 1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(bias,name=scope)
        parameters += [kernel, biases]
        # print_activations(conv3)
        # conv3   [128, 13, 13, 384]

    # 卷积层三：LRN
    with tf.name_scope('lrn3') as scope:
        lrn3 = tf.nn.local_response_normalization(conv3,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)
    # 卷积层四：卷积运算
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384],dtype=tf.float32,stddev=0.01),
                             name='weights')
        conv = tf.nn.conv2d(input=lrn3, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(bias,name=scope)
        parameters += [kernel, biases]
        # print_activations(conv4)
        # conv3[128, 13, 13, 384]

    # 卷积层四：LRN
    with tf.name_scope('lrn4') as scope:
        lrn4 = tf.nn.local_response_normalization(conv4,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # 卷积层五：卷积运算
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=0.01),
                             name='weights')
        conv = tf.nn.conv2d(input=lrn4, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                  trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        # print_activations(conv5)

    # 卷积层五：最大池化层
    with tf.name_scope('pool5') as scope:
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool5')

    return pool5, parameters
    # print_activations(pool5)
    # sess.run(tf.global_variables_initializer())
    # result = sess.run(pool5)
    # print(result.shape)

def Backward_propagation(grad):
    pass


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'   # 使用第0块GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    config.gpu_options.allocator_type = 'BFC'  # 使用BFC算法

    pool5, parameters = Forward_propagation(image)
    objective = tf.nn.l2_loss(pool5)
    grad = tf.gradients(objective,parameters)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(grad)
        print(type(result))
        print(result)