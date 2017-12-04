#from sklearn.neural_network import MLPClassifier
import tensorflow as tf

import os.path
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("/tmp/data/", one_hot=True)

from readData import *
"""
from readFaceData import loadImages

samples = loadImages()
data = faces(samples, testRatio=.3)
"""

learningRate = 1e-2
numSteps = 10000
batchSize = 50
displayStep = 1000

numInput = 28**2

# Autoencoder class {{{

class AutoEncoder:
    def __init__(self, inputSize, hiddenLayerSizes):
        encoderLayers = (inputSize,) + hiddenLayerSizes
        decoderLayers = hiddenLayerSizes[::-1] + (inputSize,)
        encoderVariables = []
        encoderBiases = []
        decoderVariables = []
        decoderBiases = []
        for i in range(len(encoderLayers) -1):
            var = tf.Variable(tf.random_normal(encoderLayers[i:i+2]))
            encoderVariables.append(var)
        for i in range(1, len(encoderLayers)):
            var = tf.Variable(tf.random_normal(encoderLayers[i:i+1]))
            encoderBiases.append(var)
        for i in range(len(decoderLayers) -1):
            var = tf.Variable(tf.random_normal(decoderLayers[i:i+2]))
            decoderVariables.append(var)
        for i in range(1, len(decoderLayers)):
            var = tf.Variable(tf.random_normal(decoderLayers[i:i+1]))
            decoderBiases.append(var)

        inputs = tf.placeholder("float", [None, inputSize])
        currentLayer = inputs
        for weight, bias in zip(encoderVariables, encoderBiases):
            currentLayer = tf.nn.sigmoid(
                    tf.add(
                        tf.matmul(currentLayer, weight),
                        bias))

        latentRepresentation = currentLayer

        for weight, bias in zip(decoderVariables, decoderBiases):
            currentLayer = tf.nn.sigmoid(
                    tf.add(
                        tf.matmul(currentLayer, weight),
                        bias))

        outputs = currentLayer
        loss = tf.reduce_mean(tf.pow(inputs - outputs, 2))
        self.inputs = inputs
        self.latentRepresentation = latentRepresentation
        self.outputs = outputs
        self.loss = loss
# }}} 

network = AutoEncoder(numInput, (256,))
X = network.inputs
optimizer = tf.train.RMSPropOptimizer(learningRate)
minimizer = optimizer.minimize(network.loss)

init = tf.global_variables_initializer()

def createImageGrid(images, gridShape=None, imageShape=None):
    """images is a 2D array where each row is an image, and the values
    in the row are all the pixels of the image unraveled into a 1D
    vector."""

    if not gridShape or not imageShape:
        raise Error

    grid = np.empty((gridShape[0] * imageShape[0],
                     gridShape[1] * imageShape[1]))
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

with tf.Session() as sess:

    sess.run(init)
    for i in range(1, numSteps + 1):
        batchX, __ = data.train.next_batch(batchSize)
        #batchX = data.train.nextBatch(batchSize)
        __, l = sess.run([minimizer, network.loss], feed_dict={X:batchX})
        if i % displayStep == 0 or i == 1:
            print("Step {}: Minibatch Loss: {}".format(i, l))

    original, __ = data.test.next_batch(20)
    modified = sess.run(network.outputs, feed_dict={X:original})

gridShape = (4,5)
imageShape = (28, 28)
originalGrid = createImageGrid(original, gridShape=gridShape,
                               imageShape=imageShape)
modifiedGrid = createImageGrid(modified, gridShape=gridShape,
                               imageShape=imageShape)
fig = plt.figure(figsize=(gridShape[0], 2 * gridShape[1]))
ax = fig.add_subplot(121)
ax.imshow(originalGrid, cmap='gray')
ax = fig.add_subplot(122)
ax.imshow(modifiedGrid, cmap='gray')

plt.show()

