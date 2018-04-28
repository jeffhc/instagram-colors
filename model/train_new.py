
from PIL import Image, ImageFilter
from pprint import pprint
import tensorflow as tf
import numpy as np
import functools
import imageio
import pickle
import sys
import piexif, piexif.helper, json
from os import listdir, mkdir
from os.path import isfile, join, exists

VGG_PATH = 'imagenet-vgg-verydeep-19.mat' # did not use
IMAGE_DATA_DIR = 'scaled_photos' # images pre-stretched to all be of same size
TOTAL_IMAGES = 1000
BATCH_SIZE = 100
LEARNING_RATE = 1e-3


def load_photos(array):
	# Takes a list of filenames, returns a dictionary with np.arrays() and metadata
	data = { 
		"images": [],
		"likes": []
	} 
	for filepath in array:
		#try:
		img = Image.open(filepath)
		img.load()
		rgb_data = np.asarray( img, dtype="int32" )
		data["images"].append(rgb_data)

		exif_dict = piexif.load(filepath)
		user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
		metadata = json.loads(user_comment)
		
		data["likes"].append(metadata["likes"])
			#all_data["ratio"].append(metadata["ratio"])
			
		#except:
		 #   print("Could not load: " + filepath)
		  #  with open('train_error_log.txt', 'a') as f:
		   #	 f.write("Could not load: " + filepath + '\n')
	return data

def chunks(array, chunk_size, mode):
	"""Yield successive n-sized chunks from l."""
	for i in range(0, len(array), chunk_size):
		data = load_photos(array[i:i + chunk_size])
		yield data[mode]

def weights_variable(shape, name):
	return tf.Variable(tf.truncated_normal(shape=shape), name=name)

def biases_variable(shape, name):
	return tf.Variable(tf.constant(0.1, shape=shape), name=name)


# Placeholder variables
images = tf.placeholder(tf.float32, shape=(None, 640, 640, 3), name='images')
likes = tf.placeholder(tf.float32, shape=(None, 1), name='likes')

### Preprocess images using VGG. Code borrowed and modified from https://github.com/lengstrom/fast-style-transfer
#images_pre = vgg.preprocess(images) # TODO: What is this doing? or rather... Is this the mean of the VGG dataset or OUR dataset?
#net = vgg.net(VGG_PATH, images_pre)
#images_vgg = net['relu5_4'] # TODO: Has the image been through the VGG_net?
#print(images_vgg.shape)

# Take last layer of net and convolve it
def general_convolution(input, layer_name):
	"Performs a 5x5 convolution on `input` with a stride length of 2 (halves input image size)"
	"Input and output channels are both 1"
	weights = weights_variable([5, 5, 3, 3], '%s_weights' % layer_name)
	convolution = tf.nn.conv2d(input, filter=weights, strides=[1, 2, 2, 1], padding='SAME')

	output_width = int(convolution.shape[1])
	output_height = int(convolution.shape[2])

	biases = biases_variable([output_width, output_height, 1], '%s_biases' % layer_name)

	return tf.nn.relu(convolution + biases)

# Convolve: After convolving, shape is (?, 40, 40, 3) --> 4800 pixels
relu1 = general_convolution(images, 'relu1')
relu2 = general_convolution(relu1, 'relu2')
relu3 = general_convolution(relu2, 'relu3')
relu4 = general_convolution(relu3, 'relu4')

# Flatten convolved image. Gives (?, 4800)
flattended_size = np.prod(relu4.shape[1:])
relu4 = tf.reshape(relu4, [-1, flattended_size])

# Fully connected network to produce number of likes
	# Image --> network
fc_weights0 = tf.Variable(tf.truncated_normal([int(flattended_size), 5000]), name='fc_weights0')
fc_biases0 = tf.Variable(tf.constant(0.1, shape=(5000,)), name='fc_biases0')
fc0 = tf.nn.relu(tf.matmul(relu4, fc_weights0) + fc_biases0)

	# network --> likes
fc_weights1 = tf.Variable(tf.truncated_normal([int(fc0.shape[1]), 1]), name='fc_weights1')
fc_biases1 = tf.Variable(tf.constant(0.1, shape=(1,)), name='fc_biases1')
output_likes = tf.matmul(fc0, fc_weights1) + fc_biases1

# Define cost
cost = tf.losses.mean_squared_error(output_likes, likes) # TODO: Does order matter?
tf.summary.scalar('cost', cost)

# Define training operation
train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.save(sess, './tensorboard/checkpoint', 0)
file_writer = tf.summary.FileWriter('logs/', sess.graph)


# Get all the filenames
all_file_names = [join(IMAGE_DATA_DIR, f) for f in listdir(IMAGE_DATA_DIR) if isfile(join(IMAGE_DATA_DIR, f))]
all_file_names = all_file_names[:TOTAL_IMAGES]

for epoch in range(100000):

	for train_images, train_likes in zip(chunks(all_file_names, BATCH_SIZE, 'images'), chunks(all_file_names, BATCH_SIZE, 'likes')):
		train_likes = np.array(train_likes)
		train_likes = np.reshape(train_likes, [len(train_likes), 1])
		sess.run(train, feed_dict={images: train_images, likes: train_likes})

	if epoch % 5 == 0:
		# Create checkpoint
		saver.save(sess, './tensorboard/checkpoint', global_step=epoch)

		# Evaluate Performance
		print('Epoch: %s of 100000' % epoch)
		print('Cost: %s' % sess.run(cost, feed_dict={images: train_images, likes: train_likes}))
		with open('cost_results.csv', 'a') as f:
			f.write(str(sess.run(cost, feed_dict={images: train_images, likes: train_likes})))
			f.write('\n')
		
		# Save results images
		results = sess.run(output_likes, feed_dict={images: train_images})

		# Save results and originals
		try:
			mkdir('results')
		except:
			pass
		with open('results/epoch_%s.txt' % epoch, 'w') as f:
			f.write(str(results) + '\n')

		# Tensorboard graphing
		merged = tf.summary.merge_all()
		summary = sess.run(merged , feed_dict={images: train_images, likes: train_likes})
		file_writer.add_summary(summary, epoch)
