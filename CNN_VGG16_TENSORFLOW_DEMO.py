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
import os

import tensorflow as tf
import time
import cifar10

# 参数(大卷积层编号,小卷积层编号,卷积核尺寸,图片矩阵，图片通道数,卷积高度步长，卷积宽度步长)
def convolution(name,kernel_size,kernel_num,image,dh_w,parameters):
    image_channel = image.get_shape().as_list()[-1]
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kernel_size[0], kernel_size[1], image_channel, kernel_num],
                                                 dtype=tf.float32,
                                                 stddev=0.1), name='weights')
        conv = tf.nn.conv2d(input=image, filter=kernel, strides=[1, dh_w[0], dh_w[1], 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[kernel_num], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(bias, name=scope)

        parameters += [kernel, biases]
        return activation

def pooling(name,input,ksize,strides,padding_method):
    with tf.name_scope(name) as scope:
        pool_result = tf.nn.max_pool(input,
                                     ksize=[1, ksize[0], ksize[1], 1],
                                     strides=[1, strides[0], strides[1], 1],
                                     padding=padding_method,
                                     name=name)
        return pool_result

def full_connection(name,input_value,output_shape,parameters):
    input_shape = input_value.get_shape().as_list()[-1]
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                 shape=[input_shape, output_shape],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[output_shape], dtype=tf.float32),
                             name='biases')
        activation = tf.nn.relu_layer(input_value, kernel, biases, name=scope)

        parameters += [kernel, biases]
        return activation



def Forward_propagation(image, keep_prob):

    '''
    前向传播
    :param image:
    :return:
    '''

    parameters = []  # 参数收集

    # 卷积层1 --> [image_num, 224, 224, 64]
    conv1 = convolution('conv1', (3, 3), 64, image, (1, 1), parameters)

    # 卷积层2 --> [image_num, 224, 224, 64]
    conv2 = convolution('conv2', (3, 3), 64, conv1, (1, 1), parameters)

    # 最大池化层a --> [image_num, 112, 112, 64]
    pool_a = pooling('pool_a', conv2, (2, 2), (2, 2), padding_method='VALID')

    # 卷积层3 --> [image_num, 112, 112, 128]
    conv3 = convolution('conv3', (3, 3), 128, pool_a, (1, 1), parameters)

    # 卷积层4 --> [image_num, 112, 112, 128]
    conv4 = convolution('conv4', (3, 3), 128, conv3, (1, 1), parameters)

    # 最大池化层b --> [image_num, 56, 56, 128]
    pool_b = pooling('pool_b', conv4, (2, 2), (2, 2), padding_method='VALID')

    # 卷积层5 --> [image_num, 56, 56, 256]
    conv5 = convolution('conv5', (3, 3), 256, pool_b, (1, 1), parameters)

    # 卷积层6 --> [image_num, 56, 56, 256]
    conv6 = convolution('conv6', (3, 3), 256, conv5, (1, 1), parameters)

    # 卷积层7 --> [image_num, 56, 56, 256]
    conv7 = convolution('conv7', (3, 3), 256, conv6, (1, 1), parameters)

    # 最大池化层c --> [image_num, 28, 28, 256]
    pool_c = pooling('pool_c', conv7, (2, 2), (2, 2), padding_method='VALID')

    # 卷积层8 --> [image_num, 28, 28, 512]
    conv8 = convolution('conv8', (3, 3), 512, pool_c, (1, 1), parameters)

    # 卷积层9 --> [image_num, 28, 28, 512]
    conv9 = convolution('conv9', (3, 3), 512, conv8, (1, 1), parameters)

    # 卷积层10 --> [image_num, 28, 28, 512]
    conv10 = convolution('conv10', (3, 3), 512, conv9, (1, 1), parameters)

    # 最大池化层d --> [image_num, 14, 14, 512]
    pool_d = pooling('pool_d', conv10, (2, 2), (2, 2), padding_method='VALID')

    # 卷积层11 --> [image_num, 14, 14, 512]
    conv11 = convolution('conv11', (3, 3), 512, pool_d, (1, 1), parameters)

    # 卷积层12 --> [image_num, 14, 14, 512]
    conv12 = convolution('conv12', (3, 3), 512, conv11, (1, 1), parameters)

    # 卷积层13 --> [image_num, 14, 14, 512]
    conv13 = convolution('conv13', (3, 3), 512, conv12, (1, 1), parameters)

    # 最大池化层e --> [image_num, 7, 7, 512]
    pool_e = pooling('pool_e', conv13, (2, 2), (2, 2), padding_method='VALID')

    # 展平层
    shape = pool_e.get_shape().as_list()
    flattend_shape = shape[1] * shape[2] * shape[3]
    flattend = tf.reshape(pool_e, [-1, flattend_shape], name='flattend')   # [1, 25088]

    # 全连接层14
    fuc14 = full_connection('full14', flattend, 2048, parameters)   #[1, 2048]
    # drop用来减轻过拟合，keep_prob在初始化时是一个占位符，后run的时候feed数值，数值为神经元随机失活概率
    fuc14_drop = tf.nn.dropout(fuc14, keep_prob, name='fuc14_drop')

    # 全连接层15
    fuc15 = full_connection('fuc15', fuc14_drop, 2018, parameters)   #[1, 2048]
    fuc15_drop = tf.nn.dropout(fuc15, keep_prob, name='fuc15_drop')


    # 全连接层16
    fuc16 = full_connection('fuc16', fuc15_drop, 2018, parameters)
    softmax = tf.nn.softmax(fuc16)
    predictions = tf.argmax(softmax, 1)

    return predictions, softmax, fuc16, parameters

def time_tensorflow_run(session, target, feed, info_string):

    start_time = time.time()
    _ = session.run(target, feed_dict = feed)
    duration = time.time() - start_time
    print(info_string+':'+str(duration))



def train_vgg16():
    with tf.Graph().as_default():
        image = tf.Variable(tf.random_normal([4, 224, 224, 3],
                                             dtype=tf.float32,
                                             stddev=0.1))
        keep_prob = tf.placeholder(tf.float32)

        predictions, softmax, fuc16, parameters = Forward_propagation(image, keep_prob)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, 'Forward_propagation')

        objective = tf.nn.l2_loss(fuc16)
        grad = tf.gradients(objective, parameters)

        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, 'Forward-backward')


if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 使用第0块GPU
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
    # config.gpu_options.allow_growth = True  # 程序按需申请内存
    # config.gpu_options.allocator_type = 'BFC'  # 使用BFC算法

    # train_vgg16()
    cifar10.maybe_download_and_extract()