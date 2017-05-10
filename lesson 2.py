import tensorflow as tf
import numpy as np

node1 = tf.constant(3.0, tf.float32)  # create a constant
node2 = tf.constant(4.0)

print(node1, node2)  ##just print the node type

sess = tf.Session()
print(sess.run(node1), sess.run(node2))  ## print the node value

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run node3:", sess.run(node3))


a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
adder_node=a+b

print(sess.run(adder_node,{a:5,b:7}))
print(sess.run(adder_node,{a:[6,9],b:[9,7]}))

c=tf.placeholder(tf.float32)

adder_and_triple= adder_node*c
print(sess.run(adder_and_triple,{a:[8,9],b:[1,6],c:3}))