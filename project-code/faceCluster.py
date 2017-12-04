""" K-Means.
Implement K-Means algorithm with TensorFlow, and apply it to classify
handwritten digit images. This example is using the MNIST database of
handwritten digits as training samples (http://yann.lecun.com/exdb/mnist/).
Note: This example requires TensorFlow v1.1.0 or over.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

from readData import *

from matplotlib import pyplot as plt

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

imagesPath = os.path.join('/', 'home', 'adam', 'Documents', 'scratch',
                          'faces')
trainImagesPath = os.path.join(imagesPath, 'train', 'output')
testImagesPath = os.path.join(imagesPath, 'test', 'output')
trainImages = readImages(trainImagesPath)
testImages = readImages(testImagesPath)
trainImages.extend(testImages)
images = trainImages
samples = np.concatenate([picToSamples(i) for i in images], axis=0)
#data = faces(samples, testRatio=.3)

full_data_x = samples

# Parameters
num_steps = 50 # Total steps to train
batch_size = 1024 # The number of samples per batch
k = 5 # The number of clusters
#num_classes = 10 # The 10 digits
num_features = samples.shape[1]

# Input images
X = tf.placeholder(tf.float32, shape=[None, num_features])
# Labels (for assigning a label to a centroid and testing)
#Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-Means Parameters
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)

# Build KMeans graph
training_graph = kmeans.training_graph()

if len(training_graph) > 6: # Tensorflow 1.4+
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph
else:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph

cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

# Training
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    #print(idx)
    #print(idx.shape)
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))

clusterIndexes = [np.arange(idx.size)[idx == i]
                  for i in np.unique(idx)]

def createImageGrid(images, gridShape=None, imageShape=None):
    """images is a 2D array where each row is an image, and the values
    in the row are all the pixels of the image unraveled into a 1D
    vector."""

    if not gridShape or not imageShape:
        raise Error

    grid = np.empty((gridShape[0] * imageShape[0], gridShape[1] * imageShape[1]))
    rows, cols = gridShape
    numImages = images.shape[0]
    for i in range(numImages):
        # images fill up each row, column-by-column.
        row = i // cols
        col = i % cols
        imageWidth, imageHeight = imageShape
        leftBound = col * imageWidth
        rightBound = (col + 1) * imageWidth
        topBound = row * imageHeight
        bottomBound = (row + 1) * imageHeight
        grid[topBound:bottomBound, leftBound:rightBound] = (
                images[i].reshape(imageShape))

    return grid

for i in range(len(clusterIndexes)):
    np.random.shuffle(clusterIndexes[i])
    grid = createImageGrid(samples[clusterIndexes[i][:40]], gridShape=(4,10),
                           imageShape=(60,60))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(grid, cmap="gray")

plt.show()
