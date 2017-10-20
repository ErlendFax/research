#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
import tensorflow as tf
import numpy as np

from server import client_generator

#def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
#    acc = sess.run(accuracy, feed_dict=feed_dict_train)
#    val_acc = sess.run(accuracy, feed_dict=feed_dict_validate)
#    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
#    print(msg.format(epoch + 1, acc, val_acc, val_loss))

#def compute_accuracy(v_xs, v_ys):
	#global prediction
	#y_pre = sess.run(prediction, feed_dict={xs: v_xs})
	#correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
	#return result

def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    X, Y, _ = tup
    Y = Y[:, -1]
    if X.shape[1] == 1:  # no temporal context
      X = X[:, -1]
    yield X, Y

def conv2d(x, W, s):
	return tf.nn.conv2d(x, W, strides=[1,s,s,1], padding="SAME")

def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variabel(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def sig2d(x):
    return tf.nn.sigmoid(x, name='Sigmoid-normalization')


if __name__ == "__main__":

  tf.reset_default_graph()
  sess = tf.Session()

  xs = tf.placeholder(tf.float32, [None, 153600])
  x_image = tf.reshape(xs, [-1, 160, 320, 3])
  ys = tf.placeholder(tf.float32, [None, 1])

  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=25, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=20, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  ch, row, col = 3, 160, 320  # camera formatrm

  x_image = tf.nn.sigmoid(x_image, name='Sigmoid-normalization')

    # conv1
  W_conv1 = weight_variable([8,8,3,16])
  b_conv1 = bias_variabel([16])
  h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1, 4) + b_conv1)

    # conv2
  W_conv2 = weight_variable([5,5,16,32])
  b_conv2 = bias_variabel([32])
  h_conv2 = tf.nn.elu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

    # conv3
  W_conv3 = weight_variable([5,5,32,64])
  b_conv3 = bias_variabel([64])
  h_conv3 = tf.nn.elu(conv2d(h_conv2, W_conv3, 2) + b_conv3)

    # flat1
  shape = h_conv3.get_shape().as_list()
  flat1 = tf.reshape(h_conv3, [-1, shape[1]*shape[2]* shape[3]])

    # drop1
  drop1 = tf.nn.dropout(flat1, 0.2)

    # elu1
  elu1 = tf.nn.elu(drop1)

    # dense1
  dense1 = tf.layers.dense(elu1, 512)

    # drop2
  drop2 = tf.nn.dropout(dense1, 0.5)

    # elu2
  elu2 = tf.nn.elu(drop2)

    # dense2
  output = tf.layers.dense(elu2, 1) # output

  with tf.name_scope("Loss"):
      loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(ys, output))))
      tf.summary.scalar("loss",loss)

  train = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())

  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter("tb-logs-no-keras/", sess.graph)


  sess.run(tf.global_variables_initializer())

  for i in range(args.epoch):
      for j in range(args.epochsize/200):
          batch_xs, batch_ys = next(gen(20, args.host, port=args.port))
          batch_val_xs, batch_val_ys = next(gen(20, args.host, port=args.val_port))
          batch_xs = np.reshape(batch_xs,(-1,153600))
          sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
          if (j % 20 == 0):
              result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys})
              writer.add_summary(result, j)
          if (j % 100 == 0):
              batch_val_xs = np.reshape(batch_xs,(-1,153600))
              val_loss = sess.run(loss, feed_dict={xs: batch_val_xs, ys: batch_val_ys})
              print "Epoch: ", '%3s' % i, " Loss: ", '%4s' % sess.run(loss, feed_dict={xs: batch_xs, ys: batch_ys}), "Val loss: ", '%4s' % val_loss
              #show_progress(i,{xs: batch_xs, ys: batch_ys},{xs: batch_val_xs, ys: batch_val_ys},val_loss)
