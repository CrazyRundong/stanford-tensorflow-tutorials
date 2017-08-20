""" The mo frills implementation of word2vec skip-gram model using NCE loss. 
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from process_data import process_data
import utils

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 64  # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 20000
SKIP_STEP = 2000  # how many steps to skip before reporting the loss


def word2vec(batch_gen):
    """ Build the graph for word2vec model and train it """
    with tf.name_scope("data"):
        # Step 1: define the placeholders for input and output
        # center_words have to be int to work on embedding lookup
        center_words = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE,), name="center_words")
        output_words = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, 1), name="target_words")

    with tf.name_scope("embed"):
        # Step 2: define weights. In word2vec, it's actually the weights that we care about
        # vocab size x embed size
        # initialized to random uniform -1 to 1
        embed_mat = tf.Variable(tf.random_uniform((VOCAB_SIZE, EMBED_SIZE), -1., 1.), name="embed_matrix")

    with tf.name_scope("loss"):
        # Step 3: define the inference
        # get the embed of input words using tf.nn.embedding_lookup
        embed = tf.nn.embedding_lookup(embed_mat, center_words, name='embed')

        # Step 4: construct variables for NCE loss
        # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
        # nce_weight (vocab size x embed size), intialized to truncated_normal stddev=1.0 / (EMBED_SIZE ** 0.5)
        # bias: vocab size, initialized to 0

        nce_weight = tf.Variable(tf.truncated_normal((VOCAB_SIZE, EMBED_SIZE), stddev=1. / (EMBED_SIZE ** 0.5)),
                                 name="nce_weight")
        nce_bias = tf.Variable(tf.zeros((VOCAB_SIZE,)),
                               name="nce_bias")
        # define loss function to be NCE loss function
        # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
        # need to get the mean accross the batch
        # note: you should use embedding of center words for inputs, not center words themselves
        loss = tf.reduce_mean(tf.nn.nce_loss(nce_weight,
                                             nce_bias,
                                             output_words,
                                             embed,
                                             num_sampled=NUM_SAMPLED,
                                             num_classes=VOCAB_SIZE))

    # Step 5: define optimizer
    opt = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    # tf.summary
    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", loss)
        tf.summary.histogram("histogram loss", loss)
        summaries_op = tf.summary.merge_all()

    # check points
    utils.make_dir("checkpoints")
    saver = tf.train.Saver()
    # hack tf-gpu windows
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        # TO DO: initialize variables
        sess.run(tf.global_variables_initializer())
        chk_pt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/checkpoint"))
        if chk_pt and chk_pt.model_checkpoint_path:
            saver.restore(sess, chk_pt.model_checkpoint_path)

        total_loss = 0.0  # we use this to calculate the average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('processed', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            centers, targets = next(batch_gen)
            # TO DO: create feed_dict, run optimizer, fetch loss_batch
            loss_batch, _, summary = sess.run([loss, opt, summaries_op],
                                     {center_words: centers, output_words: targets})
            writer.add_summary(summary, global_step=index)
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, "checkpoints/skip-gram", index)

        # write summaries
        final_embed_mat = sess.run(embed_mat)
        embed_mat_var = tf.Variable(final_embed_mat[:1000], name="embedding_output")
        sess.run(embed_mat_var.initializer)

        tb_cfg = projector.ProjectorConfig()
        embedding = tb_cfg.embeddings.add()
        embedding.tensor_name = embed_mat_var.name
        embedding.metadata_path = "processed/vocab_1000.tsv"
        summary_writer = tf.summary.FileWriter("processed")
        projector.visualize_embeddings(summary_writer, tb_cfg)

        saver_embed = tf.train.Saver((embed_mat_var,))
        saver_embed.save(sess, "processed/model.ckpt", 1)

        summary_writer.close()
        writer.close()


def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)


if __name__ == '__main__':
    main()
