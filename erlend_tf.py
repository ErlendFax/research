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

if __name__ == "__main__":

  tf.reset_default_graph()
  sess = tf.Session()

  xs = tf.placeholder(tf.float32, [None, 3,160,320])
  x_image=tf.transpose(xs, perm=[0,2,3,1])
  ys = tf.placeholder(tf.float32, [None, 1])

  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=200, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=500, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  ch, row, col = 3, 160, 320  # camera formatrm

  x_image=x_image/127.5 -1.0

    # conv1
  W_conv1 = weight_variable([8,8,3,16])
  b_conv1 = bias_variabel([16])
  h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1, 4) + b_conv1)

    # conv2
  W_conv2 = weight_variable([5,5,16,32])
  b_conv2 = bias_variabel([32])
  h_conv2 = tf.nn.elu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

  shape = h_conv2.get_shape().as_list()
  h_conv1_slice = tf.slice(h_conv1, [0, 0, 0, 0], [-1, shape[1], shape[2], 3])
  tf.summary.image("h_conv1_slice",h_conv1_slice)

    # conv3
  W_conv3 = weight_variable([5,5,32,64])
  b_conv3 = bias_variabel([64])
  h_conv3 = tf.nn.elu(conv2d(h_conv2, W_conv3, 2) + b_conv3)


  shape = h_conv2.get_shape().as_list()
  h_conv2_slice = tf.slice(h_conv2, [0, 0, 0, 0], [-1, shape[1], shape[2], 3])
  tf.summary.image("h_conv2_slice",h_conv2_slice)


    # flat1
  shape = h_conv3.get_shape().as_list()
  flat1 = tf.reshape(h_conv3, [-1, shape[1]*shape[2]* shape[3]])

  h_conv3_slice = tf.slice(h_conv3, [0, 0, 0, 0], [-1, shape[1], shape[2], 3])
  tf.summary.image("h_conv3_slice",h_conv3_slice)


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

  #SE_HER
  with tf.name_scope("Loss"):
      loss = tf.reduce_mean(tf.square(tf.subtract(ys, output)))
      tf.summary.scalar("loss",loss)

  train = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())

  sess.run(tf.global_variables_initializer())

  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter("logs/", sess.graph)

  steps_per_epoch=10000//(args.batch)

  for i in range(args.epoch):
    loss_epoch=0

    for j in range(steps_per_epoch):
      batch_xs, batch_ys = next(gen(20, args.host, port=args.port))
      tr,loss_batch=sess.run([train,loss], feed_dict={xs: batch_xs, ys: batch_ys})
      loss_epoch=loss_epoch+loss_batch

      if (j % 20 == 0):
          result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys})
          writer.add_summary(result, j)

    val_loss_epoch = 0
    for k in range(10):
      batch_xs_val, batch_ys_val = next(gen(20, args.host, port=args.port))
      val_loss=sess.run(loss, feed_dict={xs: batch_xs_val, ys: batch_ys_val})
      val_loss_epoch += val_loss


    val_loss_epoch = val_loss_epoch/10
    print ("Epoch: ", '%3s' % i, " Loss: ", '%4s' % (loss_epoch/steps_per_epoch), "Val_loss: ", '%4s' % (val_loss_epoch))

sess.close()
