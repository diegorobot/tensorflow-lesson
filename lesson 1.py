import tensorflow as tf
import numpy as np

sess = tf.Session()

W=tf.Variable([.3],tf.float32)
b=tf.Variable([-.3],tf.float32)
x=tf.placeholder(tf.float32)
linear_model=W*x+b
init=tf.global_variables_initializer()

y=tf.placeholder(tf.float32)
squared_deltas=tf.square(linear_model-y)
loss=tf.reduce_sum(squared_deltas)

optimizar =tf.train.GradientDescentOptimizer(0.01)
train=optimizar.minimize(loss)
sess.run(init)
for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    if i%20==0:
        print(i,sess.run(W),sess.run(b))
