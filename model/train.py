import vgg
from PIL import Image, ImageFilter
from pprint import pprint
import tensorflow as tf
import numpy as np
import functools
import imageio
import pickle
import os
import sys

VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
TOTAL_IMAGES = 12
BATCH_SIZE = 3
LEARNING_RATE = 1e-3

all_data = pickle.load(open('all_data.pickle', 'rb'))

def chunks(array, chunk_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(array), chunk_size):
        yield array[i:i + chunk_size]

def weights_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape=shape), name=name)

def biases_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


# Placeholder variables
images = tf.placeholder(tf.float32, shape=(None, 640, 640, 3), name='images')
likes = tf.placeholder(tf.float32, shape=(None, 1), name='likes')

### Preprocess images using VGG. Code borrowed and modified from https://github.com/lengstrom/fast-style-transfer
#images_pre = vgg.preprocess(images) # TODO: Wtf is this doing? or rather... Is this the mean of the VGG dataset or OUR dataset?
#net = vgg.net(VGG_PATH, images_pre)
#images_vgg = net['relu5_4'] # TODO: Has the image been through the VGG_net?
#print(images_vgg.shape)


# Take last layer of net and convolve it
def general_convolution(input, layer_name):
    "Performs a 5x5 convolution on `input` with a stride length of 2 (halves input image size)"
    "Input and output channels are both 1"
    weights = weights_variable([5, 5, 3, 3], '%s_weights' % layer_name)
    convolution = tf.nn.conv2d(input, filter=weights, strides=[1, 2, 2, 3], padding='SAME')

    output_width = int(convolution.shape[1])
    output_height = int(convolution.shape[2])

    biases = biases_variable([output_width, output_height, 1], '%s_biases' % layer_name)

    return tf.nn.relu(convolution + biases)

# Convolve: After convolving, shape is (40, 40, 3)
relu1 = general_convolution(images, 'relu1')
relu2 = general_convolution(relu1, 'relu2')
relu3 = general_convolution(relu2, 'relu3')
relu4 = general_convolution(relu3, 'relu4')

# Flatten convolved image. Gives (?, 4800)
flattended_size = np.prod(relu4.shape[1:])
relu4 = tf.reshape(relu4, [-1, flattended_size])

# Fully connected network to produce number of likes
fc_weights0 = tf.Variable(tf.truncated_normal([int(flattended_size), 5000]), name='fc_weights0')
fc_biases0 = tf.Variable(tf.constant(0.1, shape=(5000,)), name='fc_biases0')
fc0 = tf.nn.relu(tf.matmul(relu4, fc_weights0) + fc_biases0)

fc_weights1 = tf.Variable(tf.truncated_normal([int(fc0.shape[1]), 1]), name='fc_weights1')
fc_biases1 = tf.Variable(tf.constant(0.1, shape=(1,)), name='fc_biases1')

output_likes = tf.matmul(fc0, fc_weights1) + fc_biases1
output_likes = tf.reshape(output_likes, [-1, 1])


# Define cost
cost = tf.losses.mean_squared_error(output_likes, likes) # TODO: Does order matter?

# Define training operation
train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.save(sess, './tensorboard/checkpoint', 0)
file_writer = tf.summary.FileWriter('tensorboard/', sess.graph)


all_train_images = all_data['photos'][:TOTAL_IMAGES]
all_train_likes = all_data['likes'][:TOTAL_IMAGES]
#all_train_edge_images = np.reshape(all_data['grayedges'][:TOTAL_IMAGES], [TOTAL_IMAGES, 64, 64, 1]) # TODO: Why do we reshape this?
#all_train_content_images = all_data['original'][:TOTAL_IMAGES]

for epoch in range(100000):
    for train_images, train_likes in zip(chunks(all_train_images, BATCH_SIZE), chunks(all_train_likes, BATCH_SIZE)):
        sess.run(train, feed_dict={images: train_images, likes: train_likes})

    if epoch % 100 == 0:
        # Create checkpoint
        saver.save(sess, './tensorboard/checkpoint', global_step=epoch)

        # Evaluate Performance
        print('Epoch: %s of 100000' % epoch)
        print('Cost: %s' % sess.run(cost, feed_dict={images: train_images, likes: train_likes}))
        
        # Save results images
        results = sess.run(output_likes, feed_dict={images: train_images})
        for i in range(10): # TODO: Why is this loop here?
            result_likes = results[i]

            # Save results and originals
            try:
                os.mkdir('results/epoch%s' % epoch)
            except:
                pass
            with open('results/epoch%s/RESULT%s.txt' % (epoch, i), 'a') as f:
            	f.write(result_likes + '\n')
            with open('results/epoch%s/ORIGINAL%s.txt' % (epoch, i), 'a') as f:
            	f.write(train_likes[i] + '\n')
