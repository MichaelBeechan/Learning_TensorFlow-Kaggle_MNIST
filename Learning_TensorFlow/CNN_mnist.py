# -*- coding:utf-8 -*-
"""
Name: Michael Beechan
School: Chongqing University of Technology
Time: 2018.10.4
Description: MINIST Digit Recognizer CNN
https://www.zhihu.com/question/52668301
"""
# 卷积层1+池化层1+卷积层2+池化层2+全连接1+Dropout层+输出层
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
# Add data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Get data and deal data  astype()转换数据类型
x_train = train.iloc[:, 1:].values
x_train = x_train.astype(np.float)
x_train = np.multiply(x_train, 1.0 / 255.0)

# Get image width and height
image_size = x_train.shape[1]
images_width = images_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print('数据样本大小：(%g, %g)' % x_train.shape)
print('图像的维度大小：{0}'.format(image_size))
print('图像长度：{0}\n高度：{1}'.format(images_width, images_height))

# Get data labels
labels_flat = train.iloc[:, 0].values.ravel()
# 对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
labels_count = np.unique(labels_flat).shape[0]

# One-Hot function
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# one-hot deal labels
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('标签({0[0]}, {0[1]})'.format(labels.shape))
print('图像标签Example：[{0}] --> {1}'.format(25, labels[25]))

# Divide train data to train and validation
VALIDATION_SIZE = 2000
train_images = x_train[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

validation_images = x_train[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

# set batch size and get the sum total of batch
batch_size = 100
n_batch = len(train_images) // batch_size

# define Empty variable (data)x: 784 (labels)y: 10
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# define function to deal data
def weight_variable(shape):
    # initial weight --- normal distribution
    # 一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # initial bias -- nonzero
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# packaging TensorFlow 2D convolution
def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# packaging Tensorflow Pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Transform input data to 4D tensor, 2 and 3 is width and height, 4 is color
x_image = tf.reshape(x, [-1, 28, 28, 1])

# compute 32 features 3*3 patch
w_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])

# 28*28 images conv step-size is 1   2*2 max pool
# After pool [28/2, 28/2] = [14, 14] the second pool [14/2, 14/2] = [7, 7]
# conv data
h_conv1 = tf.nn.relu(conv2D(x_image, w_conv1) + b_conv1)
# pool result
h_pool1 = max_pool_2x2(h_conv1)

# On the previous basis, generate 64 features
w_conv2 = weight_variable([6, 6, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2D(h_pool1, w_conv2) + b_conv2)

# max_pool 2*2 --> [7, 7]
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# Fully connected neural network  1024 Neural
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 1024 to 10D output
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

# build loss function --> cross entropy
# tf.nn.softmax_cross_entropy_with_logits
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits=y_conv))
#  optimizing para
train_step_1 = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss)

# compute accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Set the filename parameter to save the model
global_step = tf.Variable(0, name='globle_step', trainable=False)
saver  =tf.train.Saver()

# initial variable
init = tf.global_variables_initializer()

# train
with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, "model.ckpt-12")
    # iter 20
    for epoch in range(1, 20):
        for batch in range(n_batch):
            # each times get one data patch to train
            batch_x = train_images[(batch) * batch_size:(batch+1) * batch_size]
            batch_y = train_labels[(batch) * batch_size:(batch+1) * batch_size]
            # the most important step -->
            sess.run(train_step_1, feed_dict={x:batch_x, y:batch_y, keep_prob:0.5})
        # each period compute accuracy
        accuracy_n = sess.run(accuracy, feed_dict={x:validation_images, y:validation_labels, keep_prob:1.0})
        print("The " + str(epoch+1) + "th, accuracy is " + str(accuracy_n))

        # save train model
        # global_step.assign(epoch).eval()
        # saver.save(sess, "model.ckpt", global_step=global_step)
