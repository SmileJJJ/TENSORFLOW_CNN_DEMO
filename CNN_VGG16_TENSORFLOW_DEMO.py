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

image = tf.Variable(tf.random_normal([1,224,224,3],
                                     dtype=tf.float32,
                                     stddev=0.1))

# 参数(大卷积层编号,小卷积层编号,卷积核尺寸,图片矩阵，图片通道数,卷积高度步长，卷积宽度步长)
def convolution(name,kernel_size,kernel_num,image,dh_w):
    image_channel = image.get_shape().as_list()[-1]
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kernel_size[0], kernel_size[1], image_channel, kernel_num],
                                                 dtype=tf.float32,
                                                 stddev=0.1),name='weights')
        conv = tf.nn.conv2d(input=image, filter=kernel, strides=[1, dh_w[0], dh_w[1], 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[kernel_num], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(bias, name=scope)

        return activation

def pooling(name,input,ksize,strides,padding_method):
    with tf.name_scope(name) as scope:
        pool_result = tf.nn.max_pool(input,
                                     ksize=[1, ksize[0], ksize[1], 1],
                                     strides=[1, strides[0], strides[1], 1],
                                     padding=padding_method,
                                     name=name)
        return pool_result

def full_connection(name,input):
    input_shape = input.get_shape().as_list()[-1]
    with tf.name_scope(name) as scope:


def Forward_propagation(image):

    '''
    前向传播
    :param image:
    :return:
    '''

    # 卷积层1 --> [image_num, 224, 224, 64]
    conv1 = convolution('conv1', (3, 3), 64, image, (1, 1))

    # 卷积层2 --> [image_num, 224, 224, 64]
    conv2 = convolution('conv2', (3, 3), 64, conv1, (1, 1))

    # 最大池化层a --> [image_num, 112, 112, 64]
    pool_a = pooling('pool_a', conv2, (2, 2), (2, 2), padding_method='VALID')

    # 卷积层3 --> [image_num, 112, 112, 128]
    conv3 = convolution('conv3', (3, 3), 128, pool_a, (1, 1))

    # 卷积层4 --> [image_num, 112, 112, 128]
    conv4 = convolution('conv4', (3, 3), 128, conv3, (1, 1))

    # 最大池化层b --> [image_num, 56, 56, 128]
    pool_b = pooling('pool_b', conv4, (2, 2), (2, 2), padding_method='VALID')

    # 卷积层5 --> [image_num, 56, 56, 256]
    conv5 = convolution('conv5', (3, 3), 256, pool_b, (1, 1))

    # 卷积层6 --> [image_num, 56, 56, 256]
    conv6 = convolution('conv6', (3, 3), 256, conv5, (1, 1))

    # 卷积层7 --> [image_num, 56, 56, 256]
    conv7 = convolution('conv7', (3, 3), 256, conv6, (1, 1))

    # 最大池化层c --> [image_num, 28, 28, 256]
    pool_c = pooling('pool_c', conv7, (2, 2), (2, 2), padding_method='VALID')

    # 卷积层8 --> [image_num, 28, 28, 512]
    conv8 = convolution('conv8', (3, 3), 512, pool_c, (1, 1))

    # 卷积层9 --> [image_num, 28, 28, 512]
    conv9 = convolution('conv9', (3, 3), 512, conv8, (1, 1))

    # 卷积层10 --> [image_num, 28, 28, 512]
    conv10 = convolution('conv10', (3, 3), 512, conv9, (1, 1))

    # 最大池化层d --> [image_num, 14, 14, 512]
    pool_d = pooling('pool_d', conv10, (2, 2), (2, 2), padding_method='VALID')

    # 卷积层11 --> [image_num, 14, 14, 512]
    conv11 = convolution('conv11', (3, 3), 512, pool_d, (1, 1))

    # 卷积层12 --> [image_num, 14, 14, 512]
    conv12 = convolution('conv12', (3, 3), 512, conv11, (1, 1))

    # 卷积层13 --> [image_num, 14, 14, 512]
    conv13 = convolution('conv13', (3, 3), 512, conv12, (1, 1))

    # 最大池化层e --> [image_num, 7, 7, 512]
    pool_e = pooling('pool_e', conv13, (2, 2), (2, 2), padding_method='VALID')

    # 展平层
    shape = pool_e.get_shape().as_list()
    flattend_shape = shape[1] * shape[2] * shape[3]
    flattend = tf.reshape(pool_e, [-1, flattend_shape], name='flattend')   # [1, 25088]
    print(flattend.get_shape().as_list())

    # 全连接层14


def train_vgg16():
    Forward_propagation(image)


if __name__ == '__main__':
    train_vgg16()