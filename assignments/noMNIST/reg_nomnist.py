# -*- coding: utf-8 -*-
# Rundong Li, ShanghaiTech

from convert_to_mnist_format import get_labels_and_files, make_arrays

import tensorflow as tf
import numpy as np

import random
import os

# hyper param for training
learning_rate = 1e-2
batch_size = 128
n_epochs = 20
nomnist_dir = "./data"
class_num = 10
train_val_ratio = 0.7
sample_num = int(batch_size * n_epochs / train_val_ratio) + 10

# build compute graph
with tf.name_scope("input_data"):
    X = tf.placeholder(tf.float32, [batch_size, 1, 28, 28], name="input_img")
    Y = tf.placeholder(tf.int32, [batch_size, 1], name="label")  # remember to encode into one-hot
    Y_one_hot = tf.one_hot(Y, class_num)

with tf.name_scope("conv_layers"):
    conv1_W = tf.random_normal((5, 5, 1, 32), stddev=0.1)
    conv1_b = tf.constant(0.1, shape=(24, 24))
    conv1 = tf.nn.conv2d(X,
                         filter=conv1_W,
                         strides=(1, 1, 1, 1),
                         padding="VALID",
                         data_format="NCHW",
                         name="conv1") + conv1_b
    relu1 = tf.nn.relu(conv1, name="relu1")
    pool1 = tf.nn.max_pool(relu1,
                           ksize=(1, 1, 2, 2),
                           strides=(1, 1, 2, 2),
                           padding="VALID",
                           data_format="NCHW",
                           name="pool1")
    flat = tf.contrib.layers.flatten(pool1)

with tf.name_scope("FC_layers"):
    W1 = tf.Variable(tf.random_normal(shape=(4608, 500), stddev=0.01), name="FC_W1")
    b1 = tf.Variable(tf.zeros(shape=(1, 500)), name="FC_b1")
    logits1 = tf.matmul(flat, W1) + b1
    logits1_relu = tf.nn.relu(logits1)

    W2 = tf.Variable(tf.random_normal(shape=(500, 10), stddev=0.01), name="FC_W2")
    b2 = tf.Variable(tf.zeros(shape=(1, 10)), name="FC_b2")
    logits = tf.matmul(logits1_relu, W2) + b2

with tf.name_scope("loss"):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot, name="loss")
    loss = tf.reduce_mean(entropy)
    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.name_scope("summary"):
    tf.summary.scalar("loss", loss)
    tf.summary.histogram("histogram_loss", loss)
    sum_op = tf.summary.merge_all()

# train

# load noMNIST data set
label_and_files = get_labels_and_files(nomnist_dir, sample_num)
random.shuffle(label_and_files)
img_array, label_array = make_arrays(label_and_files)
# expand axis of training img
img_array = np.expand_dims(img_array, axis=1)
assert img_array.shape[-3:] == (1, 28, 28), "img shape {}".format(img_array.shape)
img_train, img_val = np.split(img_array, [int(img_array.shape[0] * train_val_ratio)])
label_train, label_val = np.split(label_array, [int(label_array.shape[0] * train_val_ratio)])

tb_dir = "./processed"
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)
cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
with tf.Session(config=cfg) as sess:
    writer = tf.summary.FileWriter(tb_dir, sess.graph)
    sess.run(tf.global_variables_initializer())

    batch_idx = 0
    n_batches = int(img_train.shape[0] / batch_size / n_epochs)
    for i in range(n_epochs):
        total_loss = 0.
        for j in range(n_batches):
            X_batch, Y_batch = img_train[batch_idx: batch_idx + batch_size], label_train[batch_idx: batch_idx + batch_size]
            loss_batch, _, summary = sess.run([loss, opt, sum_op],
                                              {X: X_batch, Y: Y_batch.reshape((-1, 1))})
            writer.add_summary(summary, global_step=i + j)
            total_loss += loss_batch
            batch_idx += batch_size

        print("#{}\tloss: {}".format(i, total_loss / n_batches))

    # test model
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))  # need numpy.count_nonzero(boolarr) :(

    n_batches = int(img_val.shape[0] / batch_size)
    total_correct_preds = 0
    batch_idx = 0

    for i in range(n_batches):
        X_batch, Y_batch = img_val[batch_idx: batch_idx + batch_size], label_val[batch_idx: batch_idx + batch_size]
        accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y: Y_batch.reshape((-1, 1))})
        total_correct_preds += sum(accuracy_batch)
        batch_idx += batch_size

    print('Accuracy {0}'.format(total_correct_preds / img_val.shape[0]))

    writer.close()
