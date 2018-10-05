# -*- coding:utf-8 -*-
"""
Name: Michael Beechan
School: Chongqing University of Technology
Time: 2018.10.4
Description: tensorflow变量初始化
https://baike.baidu.com/item/TensorFlow/18828108?fr=aladdin
"""
import tensorflow as tf
# 变量定义
w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0], [1.0]])
# 矩阵乘法
y = tf.matmul(w, x)
print(y)

# 函数
norm = tf.random_normal([2, 3], mean = -1, stddev = 4)
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)  # shuffle洗牌
sess = tf.Session()
print(sess.run(norm))
print(sess.run(shuff))
# 将numpy的一些数据转换为tensorflow能用的类型
import numpy as np
a = np.zeros((3, 3))
ta = tf.convert_to_tensor(a)
print(sess.run(ta))

# 创建一个变量 并用for循环对变量进行赋值操作
num  =tf.Variable(0, name="count")
new_value = tf.add(num, 10)
op = tf.assign(num, new_value)
print(op)
# 初始化全局变量
init_op = tf.global_variables_initializer()
# 定义运行会话
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(num))
    for i in range(5):
        sess.run(op)
        print(sess.run(num))

# 通过feed设置placeholder的值
# 声明变量是不赋值，计算时进行赋值  使用feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
value_new = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(value_new, feed_dict={input1:23.0, input2:11.0}))
