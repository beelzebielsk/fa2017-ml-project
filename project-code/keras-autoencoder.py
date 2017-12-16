from keras.layers import (
        Input, Dense, Activation, Flatten
    )
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
from matplotlib import pyplot as plt

inputDim = 784
hiddenDim = 784
inputImage = Input(shape=(inputDim,))
#encoded = Dense(hiddenDim, activation='relu')(inputImage)
# The 1st argument to a Dense layer is the number of outputs of the
# layer.
encoded = Dense(hiddenDim, activation='relu', 
                activity_regularizer=regularizers.l1(1e-4))(inputImage)
decoded = Dense(inputDim, activation='sigmoid')(encoded)
autoencoder = Model(inputImage, decoded)

# Also create models out of the encoder and decoder layers.

encoderModel = Model(inputImage, encoded)
encodedInput = Input(shape=(hiddenDim,))
# Why do this and not just 'decoded'? What's the difference?
decoderLayer = autoencoder.layers[-1]
# Why say dcoderLayer(encodedInput) and not... decoded?
decoderModel = Model(encodedInput, decoderLayer(encodedInput))

# What does compile do, here? Why do we compile only this models and
# not the others? Does this model compile all the necessary stuff for
# the other models? Does "compilation" basically mean that it creates
# a tensorflow session and stores it?
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(trainSamples, __), (testSamples, __) = mnist.load_data()

trainSamples = trainSamples.astype(np.float32) / 255;
testSamples = testSamples.astype(np.float32) / 255;
trainSamples = trainSamples.reshape((len(trainSamples), np.prod(trainSamples.shape[1:])))
testSamples = testSamples.reshape((len(testSamples), np.prod(testSamples.shape[1:])))

autoencoder.fit(trainSamples, trainSamples, epochs=50, batch_size=256,
                shuffle=True, 
                validation_data = (testSamples, testSamples))

encodedImages = encoderModel.predict(testSamples)
decodedImages = decoderModel.predict(encodedImages)

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

gridSize = (4, 8)
numDigits = 4 * 8
originalDigits = testSamples[:numDigits]
predictedDigits = decodedImages[:numDigits]
originalGrid = createImageGrid(originalDigits, gridShape=gridSize,
                               imageShape=(28, 28))
predictionGrid = createImageGrid(predictedDigits, gridShape=gridSize,
                                 imageShape=(28, 28))

fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(originalGrid, cmap='gray')

ax = fig.add_subplot(122)
ax.imshow(predictionGrid, cmap='gray')

plt.show()
