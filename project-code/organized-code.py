#from sklearn.neural_network import MLPClassifier
import tensorflow as tf

import os.path
import numpy as np
from matplotlib import pyplot as plt

from readData import *
from readFaceData import loadImages

"""
imagesPath = os.path.join('/', 'home', 'adam', 'Documents', 'scratch',
                          'faces')
trainImagesPath = os.path.join(imagesPath, 'train', 'output')
testImagesPath = os.path.join(imagesPath, 'test', 'output')
trainImages = readImages(trainImagesPath)
testImages = readImages(testImagesPath)
trainImages.extend(testImages)
images = trainImages
samples = np.concatenate([picToSamples(i) for i in images], axis=0)
#print(samples.shape)
data = faces(samples, testRatio=.3)
"""

samples = loadImages()
data = faces(samples, testRatio=.3)

learningRate = 1e-8
numSteps = 1000
batchSize = 50

displayStep = 100
numHidden1 = 1024
numHidden2 = 512
numHidden3 = 256
numHidden4 = 128
numInput = samples.shape[1]

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

"""
X = tf.placeholder("float", [None, numInput])
# In each layer, the number of rows is the number of neurons in
# previous layer, and the number of columns is the number of neurons
# in the current layer.
weights = {
    'encoder1' : tf.Variable(tf.random_normal([numInput, numHidden1])),
    'encoder2' : tf.Variable(tf.random_normal([numHidden1, numHidden2])),
    'encoder3' : tf.Variable(tf.random_normal([numHidden2, numHidden3])),
    'encoder4' : tf.Variable(tf.random_normal([numHidden3, numHidden4])),
    'decoder1' : tf.Variable(tf.random_normal([numHidden4, numHidden3])),
    'decoder2' : tf.Variable(tf.random_normal([numHidden3, numHidden2])),
    'decoder3' : tf.Variable(tf.random_normal([numHidden2, numHidden1])),
    'decoder4' : tf.Variable(tf.random_normal([numHidden1, numInput])),
}

# In each layer, the number of biases is the number of neurons in the
# current layer.
biases = {
    'encoder1' : tf.Variable(tf.random_normal([numHidden1])),
    'encoder2' : tf.Variable(tf.random_normal([numHidden2])),
    'encoder3' : tf.Variable(tf.random_normal([numHidden3])),
    'encoder4' : tf.Variable(tf.random_normal([numHidden4])),
    'decoder1' : tf.Variable(tf.random_normal([numHidden3])),
    'decoder2' : tf.Variable(tf.random_normal([numHidden2])),
    'decoder3' : tf.Variable(tf.random_normal([numHidden1])),
    'decoder4' : tf.Variable(tf.random_normal([numInput])),
}

def encoder(x):
    layer1 = tf.nn.sigmoid(
            tf.add(
                tf.matmul(x, weights['encoder1']),
                biases['encoder1']))
    layer2 = tf.nn.sigmoid(
            tf.add(
                tf.matmul(layer1, weights['encoder2']),
                biases['encoder2']))
    layer3 = tf.nn.sigmoid(
            tf.add(
                tf.matmul(layer2, weights['encoder3']),
                biases['encoder3']))
    layer4 = tf.nn.sigmoid(
            tf.add(
                tf.matmul(layer3, weights['encoder4']),
                biases['encoder4']))
    return layer4

def decoder(x):
    layer1 = tf.nn.sigmoid(
            tf.add(
                tf.matmul(x, weights['decoder1']),
                biases['decoder1']))
    layer2 = tf.nn.sigmoid(
            tf.add(
                tf.matmul(layer1, weights['decoder2']),
                biases['decoder2']))
    layer3 = tf.nn.sigmoid(
            tf.add(
                tf.matmul(layer2, weights['decoder3']),
                biases['decoder3']))
    layer4 = tf.nn.sigmoid(
            tf.add(
                tf.matmul(layer3, weights['decoder4']),
                biases['decoder4']))

    return layer4

encoderOp = encoder(X)
decoderOp = decoder(encoderOp)

yPred = decoderOp
yTrue = X

# MSE
loss = tf.reduce_mean(tf.pow(yTrue - yPred, 2))
"""
network = AutoEncoder(numInput, (8000,))
X = network.inputs
optimizer = tf.train.RMSPropOptimizer(learningRate).minimize(network.loss)

init = tf.global_variables_initializer()

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

with tf.Session() as sess:

    sess.run(init)
    for i in range(1, numSteps + 1):
        batchX = data.train.nextBatch(batchSize)
        __, l = sess.run([optimizer, network.loss], feed_dict={X:batchX})
        if i % displayStep == 0 or i == 1:
            print("Step {}: Minibatch Loss: {}".format(i, l))

    original = data.test.nextBatch(20)
    modified = sess.run(network.outputs, feed_dict={X:original})

gridShape = (4,5)
imageShape = (96, 96)
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

