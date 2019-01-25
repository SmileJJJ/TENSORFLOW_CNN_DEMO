'''
测试numpy数组下的tensor计算
实现卷积层，池化层，全连接层操作
不使用TensorFlow计算图
'''

import numpy as np
import math
import tensorflow as tf

#输入数据
input_num = np.array([[0,1,1,2,4],[3,2,1,2,1],[0,0,4,1,2],[2,3,0,1,0]])
input_ndim = input_num.ndim 
input_width = input_num.shape[1]
input_height = input_num.shape[0]

#================================
#卷积层
#================================

kernel_1 = np.array([[1,1,1],[1,2,1],[1,1,1]])   #卷积核
kernel_1_width = kernel_1.shape[1]
kernel_1_height = kernel_1.shape[0]

# [[1 1 1]
#  [1 2 1]
#  [1 1 1]]

# [[0 1 1 2 4]
#  [3 2 1 2 1]
#  [0 0 4 1 2]
#  [2 3 0 1 0]]

#输入样本补零(2维)
input_num_val = np.zeros((input_height+2,input_width+2))   
input_num_val[1:input_height+1,1:input_width+1] = input_num

# [[0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 1. 2. 4. 0.]
#  [0. 3. 2. 1. 2. 1. 0.]
#  [0. 0. 0. 4. 1. 2. 0.]
#  [0. 2. 3. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0.]]

out_put_val = np.zeros((input_height,input_width))

#卷积操作
for i in range(input_height):
    for j in range(input_width):
        out_put_val[i,j] = np.sum(input_num_val[i:i+kernel_1_height,j:j+kernel_1_width]*kernel_1)

# [[ 6.  9. 10. 13. 13.]
#  [ 9. 14. 15. 20. 13.]
#  [10. 15. 18. 13.  9.]
#  [ 7. 12.  9.  9.  4.]]
print(out_put_val)
#================================
#池化层
#================================
pool_kernel_height = 2
pool_kernel_width = 2
out_put_val_height = out_put_val.shape[0]
out_put_val_width = out_put_val.shape[1]

out_put_pool_height = math.ceil(out_put_val_height/pool_kernel_height)
out_put_pool_width = math.ceil(out_put_val_width/pool_kernel_width)

out_put_pool = np.zeros((out_put_pool_height,out_put_pool_width))

for i in range(out_put_pool_height):
    for j in range(out_put_pool_width):
        out_put_pool[i,j] = np.max(out_put_val[i*pool_kernel_height:(i+1)*pool_kernel_height,j*pool_kernel_height:(j+1)*pool_kernel_height])
print(out_put_pool)

#================================
#全连接层
#================================
# [[14. 20. 13.]
#  [15. 18.  9.]]
full_connect = []
for i in out_put_pool:
    for j in i :
        full_connect.append(j)

full_connect = np.array(full_connect)
print(full_connect)


# print('kernel_1:',kernel_1)
# print('kernel_1_width:',kernel_1_width)
# print('kernel_1_height',kernel_1_height)
# print('input_ndim',input_ndim)
# print('input_width',input_width)
# print('input_height',input_height)

# print(input_num)
# print(input_num_val)
print(out_put_val)
# print(np.dot(input_num_val[0:0+kernel_1_height,0:0+kernel_1_width],kernel_1))
# print(out_put_val)