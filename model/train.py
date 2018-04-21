from utils import get_img
import vgg
from PIL import Image, ImageFilter
from pprint import pprint
import tensorflow as tf
import numpy as np
import transform
import functools
import imageio
import pickle
import os


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


### Preprocess images using VGG. Code borrowed and modified from https://github.com/lengstrom/fast-style-transfer

images = tf.placeholder(tf.float32, shape=(None, 64, 64, 3), name='images')

images_pre = vgg.preprocess(images)
net = vgg.net(VGG_PATH, images_pre)

# Fully connected network to produce output image
images_pre_size = int(images_pre.shape[1])
fc_weights0 = tf.Variable(tf.truncated_normal([images_pre_size, 5000]), name='fc_weights0')
fc_biases0 = tf.Variable(tf.constant(0.1, shape=(5000,)), name='fc_biases0')
fc0 = tf.nn.relu(tf.matmul(all_image_information, fc_weights0) + fc_biases0)

fc_weights1 = tf.Variable(tf.truncated_normal([int(fc0.shape[1]), 64*64*3]), name='fc_weights1')
fc_biases1 = tf.Variable(tf.constant(0.1, shape=(64*64*3,)), name='fc_biases1')

output_images = tf.matmul(fc0, fc_weights1) + fc_biases1
output_images = tf.reshape(output_images, [-1, 64, 64, 3])

# Define cost
cost = tf.losses.mean_squared_error(images, output_images)

# Define training operation
train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.save(sess, './tensorboard/checkpoint', 0)
file_writer = tf.summary.FileWriter('tensorboard/', sess.graph)

all_train_edge_images = np.reshape(all_data['grayedges'][:TOTAL_IMAGES], [TOTAL_IMAGES, 64, 64, 1])
all_train_content_images = all_data['original'][:TOTAL_IMAGES]

for epoch in range(100000):
    for train_edge_images, train_content_images in zip(chunks(all_train_edge_images, BATCH_SIZE), chunks(all_train_content_images, BATCH_SIZE)):
        sess.run(train, feed_dict={edge_images: train_edge_images, style_images: train_content_images, original_images: train_content_images})

    if epoch % 100 == 0:
        # Create checkpoint
        saver.save(sess, './tensorboard/checkpoint', global_step=epoch)

        # Evaluate Performance
        print('Epoch: %s of 100000' % epoch)
        print('Cost: %s' % sess.run(cost, feed_dict={edge_images: train_edge_images, style_images: train_content_images, original_images: train_content_images}))
        
        # Save results images
        results = sess.run(output_images, feed_dict={edge_images: train_edge_images, style_images: train_content_images})
        for i in range(10):
            # Scale images to [0, 255]
            result_image = results[i]
            result_image = result_image + (-np.min(result_image))
            result_image = (result_image/np.max(result_image))*255

            # Save results and originals
            try:
                os.mkdir('results/epoch%s' % epoch)
            except:
                pass
            imageio.imwrite('results/epoch%s/RESULT%s.png' % (epoch, i), result_image)
            imageio.imwrite('results/epoch%s/ORIGINAL%s.png' % (epoch, i), train_content_images[i])
