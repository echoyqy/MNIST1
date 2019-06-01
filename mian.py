# import tensorflow as tf
# import tensorflow.examples.tutorials.mnist.input_data as input_data
# # 实现图像可视化
# import matplotlib.pyplot as plt
#
# mnist=input_data.read_data_sets('MNIST_data/', one_hot=True)
# print('train的纬度：', mnist.train.images.shape)
#
#
# def plot_imag(imag):
#   plt.imshow(imag.reshape(28, 28), cmap='binary')
#   plt.show()
# # 实现图像可视化
# plot_imag(mnist.train.images[19000])

import tensorflow as tf
import urllib
import  tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)



#建立BP神经网络模型
num_classes = 10#数据类型0-9
input_size = 784#28*28
hidden_units_size = 30#层节点数
batch_size = 100#
training_iterations = 5000#迭代次数
learning_rate=0.1
# 学习速率
# 设置变量
X = tf.placeholder(tf.float32, shape = [None, input_size])
Y = tf.placeholder(tf.float32, shape = [None, num_classes])
W1 = tf.Variable(tf.random_normal ([input_size, hidden_units_size],
                                    stddev = 0.1))#hidden_units_size = 30#正态分布随机数
B1 = tf.Variable(tf.constant (0.1),
                  [hidden_units_size])#常数为1，形状为（1,1）
W2 = tf.Variable (tf.random_normal ([hidden_units_size,
                                     num_classes], stddev = 0.1))#正态分布随机数
B2 = tf.Variable (tf.constant (0.1), [num_classes])
# 搭建计算网络 使用 relu 函数作为激励函数 这个函数就是 y = max (0,x) 的一个类似线性函数 拟合程度还是不错的
# 使用交叉熵损失函数 这是分类问题例如 ： 神经网络 对率回归经常使用的一个损失函数
#第1层神经网络
hidden_opt = tf.matmul (X, W1) + B1#矩阵运算
hidden_opt = tf.nn.relu (hidden_opt)#激活函数
#第2层神经网络
final_opt = tf.matmul (hidden_opt, W2) + B2#矩阵运算
final_opt = tf.nn.relu (final_opt)#激活函数,最终的输出结果
loss = tf.reduce_mean (
    tf.nn.softmax_cross_entropy_with_logits (labels = Y, logits = final_opt))#损失函数,交叉熵方法
opt = tf.train.GradientDescentOptimizer (learning_rate).minimize (loss)
init = tf.global_variables_initializer ()#全局变量初始化
correct_prediction = tf.equal (tf.argmax (Y, 1), tf.argmax (final_opt, 1))
accuracy = tf.reduce_mean (tf.cast (correct_prediction, 'float'))#将张量转化成float


# 建立深度学习网络
sess=tf.Session()
sess.run(init)
for iter in range(training_iterations):
    batch = mnist.train.next_batch(batch_size)  # 每次迭代选用的样本数100
    batch_input = batch[0]
    batch_labels = batch[1]
    loss_value= sess.run([opt, loss ], feed_dict={X: batch_input, Y: batch_labels})
    if (iter+1) % 1000 == 0 :
        train_accuracy = accuracy.eval (session = sess, feed_dict = {X: batch_input,Y: batch_labels})
        print ("step : %d, training accuracy = %g " % (iter+1, train_accuracy))

