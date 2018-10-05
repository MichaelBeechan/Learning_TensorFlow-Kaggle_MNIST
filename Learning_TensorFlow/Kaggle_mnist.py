# -*- coding:utf-8 -*-
"""
Name: Michael Beechan
School: Chongqing University of Technology
Time: 2018.10.4
Description: Kaggle MINIST 手写图片识别  Digit Recognizer
http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html
"""
"""
一、数据的准备
二、模型的设计
三、代码实现
28*28 = 784 的二维数组
训练数据和测试数据，都可以分别转化为[42000,769]和[28000,768]的数组
模型建立：
1）使用一个最简单的单层的神经网络进行学习
2）用SoftMax来做为激活函数
3）用交叉熵来做损失函数
4）用梯度下降来做优化方式
"""
### 88.45% 识别正确率
import pandas as pd
import numpy as np
import tensorflow as tf

# 加载数据
train = pd.read_csv("train.csv")
images = train.iloc[:, 1:].values
#labels_flat = train[[0]].values.ravel()
labels_flat = train.iloc[:, 0].values.ravel()
# 输入处理
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
print("输入数据的数量：(%g, %g)" % images.shape)
images_size = images.shape[1]
images_width = images_height = np.ceil(np.sqrt(images_size)).astype(np.uint8)
print("图片的长 = {0}\n图片的高 = {1}".format(images_width, images_height))

x = tf.placeholder('float', shape=[None, images_size])
# 结果处理
labels_count = np.unique(labels_flat).shape[0]
print('结果的种类 = {0}'.format(labels_count))
y = tf.placeholder('float', shape=[None, labels_count])
# One-Hot编码 :离散特征处理——独热编码  scikit_learn有封装了现成的编码函数OneHotEncoder()
def dense_to_one_hot(labels_dense, num_calsses):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_calsses
    labels_one_hot = np.zeros((num_labels, num_calsses))
    #flat返回的是一个迭代器
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)
print('结果的数量：({0[0]}, {0[1]})'.format(labels.shape))
# 数据划分
VALIDATION_SIZE = 2000

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

batch_size = 100
n_batch = len(train_images)//batch_size

# 建立神经网络
weight = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))
result = tf.matmul(x, weight) + biases
prediction = tf.nn.softmax(result)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):
        for batch in range(n_batch):
            batch_x = train_images[batch * batch_size:(batch+1) * batch_size]
            batch_y = train_labels[batch * batch_size:(batch+1) * batch_size]
            sess.run(train_step, feed_dict={x:batch_x, y:batch_y})
        accuracy_n = sess.run(accuracy, feed_dict={x:validation_images, y:validation_labels})
        print("第"+str(epoch+1)+"轮，准确度为：" + str(accuracy_n))