from keras.layers import (
        Input, Dense, Activation, Flatten, Dropout,
    )
from keras.models import (Model, Sequential)
from keras.datasets import mnist
from keras import regularizers

import numpy as np
from matplotlib import pyplot as plt
import json

inputDim = 784
outputDim = inputDim
inputImage = Input(shape=(inputDim,))

networkCharacteristics = {
    'layers' : [
        {
            'units' : 512,
            'activation' : 'relu',
            #'activity_regularizer' : regularizers.l1(1e-5),
            'input_shape' : (inputDim, ),
        },
        {
            'units' : 256,
            #'activity_regularizer' : regularizers.l1(1e-5),
            'activation' : 'relu',
        },
        {
            'units' : 128,
            'activity_regularizer' : regularizers.l1(1e-5),
            'activation' : 'relu',
        },
        {
            'units' : 256,
            #'activity_regularizer' : regularizers.l1(1e-6),
            'activation' : 'relu',
        },
        {
            'units' : 512,
            #'activity_regularizer' : regularizers.l1(1e-6),
            'activation' : 'relu',
        },
        {
            'units' : outputDim,
            #'activity_regularizer' : regularizers.l1(1e-6),
            'activation' : 'sigmoid',
        },
    ],
    'model' : {
        'optimizer' : 'adadelta',
        'loss' : 'binary_crossentropy',
        'metrics' : ['accuracy'],
    },
    'fit' : {
        'epochs' : 20,
        'batch_size' : 256,
        'shuffle' : True,
    },
}

"""
layerList = [Dense(**layer)
              for layer in networkCharacteristics['layers']]
"""
layerList = [
    Dense(
        units=1000,
        activation='relu',
        input_shape=(inputDim,),
        activity_regularizer=regularizers.l1(1e-6),
    ),
    Dense(
        units=1200,
        activation='relu',
        activity_regularizer=regularizers.l1(1e-6),
    ),
    Dense(
        units=1000,
        activation='relu',
        activity_regularizer=regularizers.l1(1e-6),
    ),
    Dense(
        units=outputDim,
        activation='relu',
        activity_regularizer=regularizers.l1(1e-6),
    ),
]

# The number of images is not specified in the input shape because
# everything works the same regardless of this shape. So for the input
# shape, just specify the shape of each piece of info, not of the
# whole batch.
# The 1st parameter to a layer is the size of the output of that
# layer.
autoencoder = Sequential(layerList)

# This specifies how the model learns: the optimizer and loss
# function, as well as what metrics to report.
autoencoder.compile(**networkCharacteristics['model'])
                    

(trainSamples, __), (testSamples, __) = mnist.load_data()

# Normalize the input data.
trainSamples = trainSamples.astype(np.float32) / 255;
testSamples = testSamples.astype(np.float32) / 255;
trainSamples = trainSamples.reshape(
        (len(trainSamples), np.prod(trainSamples.shape[1:])))
testSamples = testSamples.reshape(
        (len(testSamples), np.prod(testSamples.shape[1:])))
# Verbose=2 for getting output stats without getting progress bar.
history = \
autoencoder.fit(trainSamples, trainSamples,
                validation_data=(testSamples, testSamples), verbose=2,
                **networkCharacteristics['fit'])

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

def stringifyActivityRegularizer(regularizer):
    if regularizer is None:
        return None
    elif isinstance(regularizer, dict):
        if regularizer['class_name'] == 'L1L2':
            return "L1L2({l1:.0e}, {l2})".format(**regularizer['config'])
    else:
        return regularizer

def displayResults(testData, model, imageShape=None,
                   gridShape=(4,8), tables=None, tableParams=dict()):
    """tables is a dictionary which specifies the form of each table.
    It takes keyword parameters for axes.table method of matplotlib.
    Each key is the name of a table (name not important right now) and
    each value is a dict which contains keyword parameters to the
    matplotlib.pyplot.axes.table function."""

    if imageShape is None:
        raise ValueError("imageShape cannot be 'None'!")
    elif tables is None:
        raise ValueError("tables cannot be 'None'!")

    numDigits = gridShape[0] * gridShape[1]
    originalDigits = testData[:numDigits]
    predictedDigits = model.predict(testSamples[:numDigits])

    originalGrid = createImageGrid(originalDigits, gridShape=gridShape,
                                   imageShape=imageShape)
    predictionGrid = createImageGrid(predictedDigits, gridShape=gridShape,
                                    imageShape=imageShape)

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.imshow(originalGrid, cmap='gray')
    ax.axis('off')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.set_title("Original Images")

    ax = fig.add_subplot(222)
    ax.imshow(predictionGrid, cmap='gray')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.axis('off')
    ax.set_title("Images Encoded then Decoded")

    ax = fig.add_subplot(223)
    ax.axis('off')
    layerTable = ax.table(**tables['layer'], **tableParams)

    ax = fig.add_subplot(224)
    ax.axis('off')
    modelTable = ax.table(**tables['model'], **tableParams)
    performanceTable = ax.table(
            **tables['performance'], **tableParams)

    layerTable.auto_set_font_size(False)
    layerTable.set_fontsize(16)
    modelTable.auto_set_font_size(False)
    modelTable.set_fontsize(16)
    performanceTable.auto_set_font_size(False)
    performanceTable.set_fontsize(16)
    plt.show()


gridShape = (4, 8)
numDigits = gridShape[0] * gridShape[1]
originalDigits = testSamples[:numDigits]
predictedDigits = autoencoder.predict(testSamples[:numDigits])
originalGrid = createImageGrid(originalDigits, gridShape=gridShape,
                               imageShape=(28, 28))
predictionGrid = createImageGrid(predictedDigits, gridShape=gridShape,
                                 imageShape=(28, 28))

fig = plt.figure()
ax = fig.add_subplot(321)
ax.imshow(originalGrid, cmap='gray')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title("Original Images")

ax = fig.add_subplot(322)
ax.imshow(predictionGrid, cmap='gray')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title("Images Encoded then Decoded")

modelConf = autoencoder.get_config()

def cleanConf(modelConf):
    # Return a copy of the config that satisfies these constraints.
    clean = {}
    # Tests which specify parts of a keras configuration that I don't
    # want to keep in the printed configuration.
    tests = [
        lambda k,v: True if v is None else False,
        lambda k,v: True if k == 'name' else False,
        lambda k,v: True if k == 'trainable' else False,
        lambda k,v: True if k == 'kernel_initializer' else False,
        lambda k,v: True if 'bias' in k else False,
        lambda k,v: True if k in ['l1', 'l2'] and v == 0.0 else False,
        lambda k,v: True if k == 'dtype' else False,
        lambda k,v: True if k == 'batch_input_shape' else False,
    ]
    if isinstance(modelConf, (str, int, float, bool)):
        return modelConf
    if isinstance(modelConf, dict):
        for key, value in modelConf.items():
            results = [t(key, value) for t in tests]
            if not any(results):
                clean[key] = cleanConf(value)
        return clean
    elif isinstance(modelConf, (list, tuple)):
        return [cleanConf(x) for x in modelConf]

def stringifyModelConf(modelConf):
    entries = [json.dumps(x) for x in modelConf]
    return '\n'.join(entries)

ax = fig.add_subplot(325)
ax.axis('off');
ax.text(0,0, stringifyModelConf(cleanConf(modelConf)), fontsize=14)

"""
ax = fig.add_subplot(223)
ax.axis('off')
# Easier to get per-layer characteristics from Keras than to manually
# maintain them in dict.


# Layer Table
rowLabels = ['Layer ' + str(i) for i in range(1, len(modelConf) + 1)]
#colLabels = ['units', 'activation', 'activity_regularizer']
colLabels = ['units', 'Layer', 'Activation', 'regularization']
getConfVals = [
    lambda conf: conf['config']['units'],
    lambda conf: conf['class_name'],
    lambda conf: conf['config']['activation'],
    lambda conf: stringifyActivityRegularizer(
                    conf['config']['activity_regularizer']),
]

layerCharacteristics = [[get(layerConf) for get in getConfVals]
                        for layerConf in modelConf]
cellText=layerCharacteristics

layerTable = ax.table(cellText=cellText, rowLabels=rowLabels,
                 colLabels=colLabels, loc='upper right', colWidths=[.2, .2, .3, .3])
"""

ax = fig.add_subplot(324)
ax.axis('off')
rowLabels = list(networkCharacteristics['model'].keys())
rowLabels.extend(list(networkCharacteristics['fit'].keys()))
cellText = [[v] for v in networkCharacteristics['model'].values()]
cellText.extend([[v] for v in networkCharacteristics['fit'].values()])
modelTable = ax.table(cellText=cellText, rowLabels=rowLabels,
                      loc='upper left', colWidths=[.8])


ax = fig.add_subplot(323)
ax.axis('off')
rowLabels = autoencoder.metrics_names
cellText = [[v] for v in autoencoder.test_on_batch(
                originalDigits, originalDigits)]
performanceTable = ax.table(cellText=cellText, rowLabels=rowLabels,
                            loc='center', colWidths=[.5])

#layerTable.auto_set_font_size(False)
#layerTable.set_fontsize(14)
modelTable.auto_set_font_size(False)
modelTable.set_fontsize(14)
performanceTable.auto_set_font_size(False)
performanceTable.set_fontsize(14)
plt.show()
