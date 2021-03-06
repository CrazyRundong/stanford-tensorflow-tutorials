""" An implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow.

Author: Chip Huyen (huyenn@stanford.edu)
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
For more details, please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import tensorflow as tf
from PIL import Image

import vgg_model
import utils

# parameters to manage experiments
STYLE = 'starry_night'
CONTENT = 'rundong_teddy-kelley'
STYLE_IMAGE = 'styles/' + STYLE + '.jpg'
CONTENT_IMAGE = 'content/' + CONTENT + '.jpg'
with Image.open(CONTENT_IMAGE) as content_img:
    style_img = Image.open(STYLE_IMAGE)
    ws, hs = style_img.size
    wc, hc = content_img.size
    w = min(ws, wc)
    h = min(hs, hc)
    del style_img
RESIZE_RATIO = 1.
IMAGE_HEIGHT = int(h * RESIZE_RATIO)
IMAGE_WIDTH = int(w * RESIZE_RATIO)
NOISE_RATIO = 0.6 # percentage of weight of the noise for intermixing with the content image

# Layers used for style features. You can change this.
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
W = [0.5, 1.0, 1.5, 3.0, 4.0] # give more weights to deeper layers.

# Layer used for content features. You can change this.
CONTENT_LAYER = 'conv4_2'

ITERS = 300
LR = 2.0

SAVE_EVERY = 20

MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
""" MEAN_PIXELS is defined according to description on their github:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8
'In the paper, the model is denoted as the configuration D trained with scale jittering. 
The input images should be zero-centered by mean pixel (rather than mean image) subtraction. 
Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].'
"""

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783

def _create_content_loss(p, f):
    """ Calculate the loss between the feature representation of the
    content image and the generated image.
    
    Inputs: 
        p, f are just P, F in the paper 
        (read the assignment handout if you're confused)
        Note: we won't use the coefficient 0.5 as defined in the paper
        but the coefficient as defined in the assignment handout.
        in this implementation, the coefficient is
        1 / ( 4 * mul(p.shape))
    Output:
        the content loss

    """
    s = tf.cumprod(p.shape)[-1]
    err = tf.norm(p - f) / tf.to_float(4 * s)

    return err


def _gram_matrix(F, N, M):
    """ Create and return the gram matrix for tensor F

    Inputs:
        N: num of filters
        M: H * W of feature map
        F: feature maps \in \mathbb{R}^{batch_size * H * W * num_channel}
        Hint: you'll first have to reshape F
    """
    F_flatten = tf.reshape(F, (1, M, N))
    # G = np.zeros((N, N), dtype=np.float32)
    # for i in range(N):
    #     for j in range(N):
    #         G[i, j] = tf.matmul(F_flatten[0, :, i], F_flatten[0, :, j], transpose_b=True).eval()
    G = tf.matmul(F_flatten, F_flatten, transpose_a=True)

    return G

def _single_style_loss(a, g):
    """ Calculate the style loss at a certain layer
    Inputs:
        a is the feature representation of the real image
        g is the feature representation of the generated image
    Output:
        the style loss at a certain layer (which is E_l in the paper)

    Hint: 1. you'll have to use the function _gram_matrix()
        2. we'll use the same coefficient for style loss as in the paper
        3. a and g are feature representation, not gram matrices
    """
    h, w, c = a.shape[-3:]
    N = c
    M = h * w
    A = _gram_matrix(a, N, M)
    G = _gram_matrix(g, N, M)

    return tf.norm(A - G) / tf.to_float(4 * N * N * M * M)

def _create_style_loss(A, model):
    """ Return the total style loss
    """
    n_layers = len(STYLE_LAYERS)
    E = [_single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]
    
    ###############################
    ## TODO: assign different weight to each layer
    return tf.reduce_sum([e * w for e, w in zip(E, W)])
    ###############################

def _create_losses(model, input_image, content_image, style_image):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image)) # assign content image to the input variable
            p = sess.run(model[CONTENT_LAYER])
        content_loss = _create_content_loss(p, model[CONTENT_LAYER])

        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in STYLE_LAYERS])                              
        style_loss = _create_style_loss(A, model)

        ##########################################
        ## TODO: create total loss.
        ## Hint: don't forget the content loss and style loss weights
        alpha = 1. / 1000.  # weight of content
        beta = 1.  # weight of style
        total_loss = alpha * content_loss + beta * style_loss
        ##########################################

    return content_loss, style_loss, total_loss

def _create_summary(model):
    """ Create summary ops necessary
        Hint: don't forget to merge them
    """
    with tf.variable_scope("summary"):
        tf.summary.scalar("content_loss", model["content_loss"])
        tf.summary.scalar("style_loss", model["style_loss"])
        tf.summary.scalar("total_loss", model["total_loss"])
        sum_op = tf.summary.merge_all()

    return sum_op

def train(model, generated_image, initial_image):
    """ Train your model.
    Don't forget to create folders for checkpoints and outputs.
    """
    skip_step = 1
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        saver = tf.train.Saver()
        ###############################
        ## TODO:
        ## 1. initialize your variables
        sess.run(tf.global_variables_initializer())
        ## 2. create writer to write your graph
        writer = tf.summary.FileWriter("summary", sess.graph)
        ###############################
        sess.run(generated_image.assign(initial_image))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = model['global_step'].eval()

        # generated image tensor
        start_time = time.time()
        for index in range(initial_step, ITERS):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 20
            
            sess.run(model['optimizer'])
            if (index + 1) % skip_step == 0:
                ###############################
                ## TODO: obtain generated image and loss
                gen_image, summary, total_loss = sess.run([generated_image,
                                                           model["summary_op"],
                                                           model["total_loss"]])
                ###############################
                gen_image = gen_image + MEAN_PIXELS
                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Time: {}'.format(time.time() - start_time))
                start_time = time.time()

                filename = 'outputs/%d.png' % (index)
                utils.save_image(filename, gen_image)

                if (index + 1) % SAVE_EVERY == 0:
                    saver.save(sess, 'checkpoints/style_transfer', index)

def main():
    with tf.variable_scope('input') as scope:
        # use variable instead of placeholder because we're training the intial image to make it
        # look like both the content image and the style image
        input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=tf.float32)
    
    utils.download(VGG_DOWNLOAD_LINK, VGG_MODEL, EXPECTED_BYTES)
    utils.make_dir('checkpoints')
    utils.make_dir('outputs')
    model = vgg_model.load_vgg(VGG_MODEL, input_image)
    model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    
    content_image = utils.get_resized_image(CONTENT_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    content_image = content_image - MEAN_PIXELS
    style_image = utils.get_resized_image(STYLE_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    style_image = style_image - MEAN_PIXELS

    model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model, 
                                                    input_image, content_image, style_image)
    ###############################
    ## TODO: create optimizer
    # and record global step
    model['optimizer'] = tf.train.AdamOptimizer(learning_rate=LR).minimize(model['total_loss'],
                                                                           global_step=model["global_step"])
    ###############################
    model['summary_op'] = _create_summary(model)

    initial_image = utils.generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, NOISE_RATIO)
    train(model, input_image, initial_image)

if __name__ == '__main__':
    main()
