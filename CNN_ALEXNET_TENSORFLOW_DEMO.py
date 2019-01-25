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

#TensorFlow里要求输入数据的shape为[图片数量，图片高度，图片宽度，图片通道]
images = tf.Variable(tf.random_normal([128, 227, 227, 3],
                                      dtype=tf.float32,
                                      stddev=1e-1))

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())

with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(input=images,filter=kernel,strides=[1,4,4,1],padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv,biases)
    conv1 = tf.nn.relu(bias,name=scope)


with tf.name_scope('lrn') as scope:
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

with tf.name_scope('pool1') as scope:
    pool1 = tf.nn.max_pool(lrn1,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1],
                          padding='VALID',
                          name='pool1')
    # print_activations(pool1)
    #pool1/pool1   [128, 27, 27, 96]

with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=0.01),
                         name='weights')
    conv = tf.nn.conv2d(input=pool1, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),
                         trainable=True,name='biases')
    bias = tf.nn.bias_add(conv,biases)
    conv2 = tf.nn.relu(bias,name=scope)
    print_activations(conv2)
    #conv2   [128, 27, 27, 256]

with tf.name_scope('lrn2') as scope:
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

with tf.name_scope('pool2') as scope:
    pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID')
    print_activations(pool2)

with tf.name_scope

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     cnn_result = sess.run(pool1)
#     print(cnn_result.shape)