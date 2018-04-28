import PIL.Image
from tkinter import *
from tkinter import filedialog
from pprint import pprint
import tensorflow as tf
import numpy as np
import functools
import imageio
import pickle
import sys
import math
import piexif, piexif.helper, json
from os import listdir, mkdir
from os.path import isfile, join, exists

INTENDED_SIZE = (640, 640)
LEARNING_RATE = 1e-3 

def load_photo(filepath):
	# Takes a filename, returns an array of 1 np.array()
	data = []
	
	img = PIL.Image.open(filepath)
	img.load()
	if (img.size) != INTENDED_SIZE:
		img = img.resize(INTENDED_SIZE, PIL.Image.ANTIALIAS)
	rgb_data = np.asarray( img, dtype="int32" )
	data.append(rgb_data)

	return data

def weights_variable(shape, name):
	return tf.Variable(tf.truncated_normal(shape=shape), name=name)

def biases_variable(shape, name):
	return tf.Variable(tf.constant(0.1, shape=shape), name=name)

# Clear the graph
tf.reset_default_graph()

# Placeholder variables
images = tf.placeholder(tf.float32, shape=(None, 640, 640, 3), name='images')
likes = tf.placeholder(tf.float32, shape=(None, 1), name='likes')

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

# Define training operation
train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)


sess = tf.Session()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
saver.restore(sess, "./project_checkpoint/checkpoint-2945")
print("Model restored.")

# Setting up Tkinter App
window = Tk()
window.title("Instagram Likes Predictor")
window.geometry('315x200')


# Functions
def clicked():
	filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
	print(filename)
	model_estimate = sess.run(output_likes, feed_dict={images:load_photo(filename)})[0][0]
	print(model_estimate)
	lbl.configure(text='Like Prediction: %d' % model_estimate)

# Tkinter GUI Items
lbl = Label(window, text="Like Prediction: ", font=("Arial Bold", 16))
lbl.pack(fill='x')
lbl2 = Label(window, text="This model predicts number of likes on photos for Instagram users who have between 100k and 200k followers on Instagram.", font=("Arial", 12), wraplength=200)
lbl2.pack(fill='x')
btn = Button(window, text="Import Photo", bg="orange", fg="blue", command=clicked)
btn.pack(side='top')
 
window.mainloop()