import glob
import os.path
import skimage.io
import numpy as np

def readImages(directoryPath):
    files = glob.glob(os.path.join(directoryPath, '*.png'))
    pics = [skimage.io.imread(f) for f in files]
    return pics

# Each picture is a sample, not a pixel. Each pixel is a feature of a
# sample.
def picToSamples(pic):
    # Assume that picture is 2D.
    is2DArr = len(pic.shape) == 2
    is3DArr = len(pic.shape) == 3
    isGreyscale = is2DArr or (is3DArr and pic.shape[2] == 1)
    isColor = is3DArr and pic.shape[2] in [3,4]
    if is2DArr:
        width, length = pic.shape
        samples = pic.reshape(1, (width*length))
    elif is3DArr:
        width, length, channels = pic.shape[:3]
        samples = pic.reshape(1, (width*length*channels))
    return samples

class batch:
    def __init__(self, images):
        self.images = images
        self.batchIndexes_ = None

    def next_batch(self, amount):
        if self.batchIndexes_ is None:
            numImages = self.images.shape[0]
            self.batchIndexes_ = np.arange(numImages)
            np.random.shuffle(self.batchIndexes_)
        if self.batchIndexes_.size < amount:
            numImages = self.images.shape[0]
            newIndexes = np.arange(numImages)
            np.random.shuffle(newIndexes)
            self.batchIndexes_ = np.concatenate([self.batchIndexes_, newIndexes], axis=0)
        #print(self.batchIndexes_)
        #print(self.batchIndexes_.dtype)
        batchImages = self.batchIndexes_[:amount]
        #print(batchImages)
        #print(type(batchImages))
        #print(batchImages.shape)
        self.batchIndexes_ = self.batchIndexes_[amount:]
        # To keep with interface of mnist data set from tensorflow.
        return (self.images[batchImages, :], None)

class faces:
    def __init__(self, images, testRatio=.5):
        numTest = int(images.shape[0] * testRatio)
        self.test = batch(images[:numTest])
        self.train = batch(images[numTest:])

def loadMnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("/tmp/data/", one_hot=True)

def loadFace(path):
    faceStuff = np.load(path)
    # Of shape (imageHeight, imageWidth, images)
    data = faceStuff['face_images']
    height, weight, images = data.shape

    # I want to reshape to (images, imageHeight*imageWidth)
    data = np.rollaxis(data, 2, 0)
    return faces(data.reshape(images, height*weight), testRatio=.3)

dataCharacteristics = {
    'mnist' : {
        'name' : 'mnist-digits',
        'photoResolution' : (28, 28),
        'numInputs' : 784,
        'load' : loadMnist
    },
    'faces' : {
        'name' : 'faces',
        'photoResolution' : (96, 96),
        'numInputs' : 9216,
        'path' : "/home/adam/Documents/scratch/faces/face_images.npz"
    },
    'smallFaces' : {
        'name' : 'faces 60x60',
        'photoResolution' : (60, 60),
        'numInputs' : 3600,
        'path' : "/home/adam/Documents/scratch/faces/small_face_images.npz"
    },
    'reallySmallFaces' : {
        'name' : 'faces 28x28',
        'photoResolution' : (28, 28),
        'numInputs' : 784,
        'path' : "/home/adam/Documents/scratch/faces/really_small_face_images.npz"
    },
}

dataCharacteristics['faces']['load'] = \
    lambda: loadFace(dataCharacteristics['faces']['path'])
dataCharacteristics['smallFaces']['load'] = \
    lambda: loadFace(dataCharacteristics['smallFaces']['path'])
dataCharacteristics['reallySmallFaces']['load'] = \
    lambda: loadFace(dataCharacteristics['reallySmallFaces']['path'])

def dictCopy(toCopy):
    copy = {}
    primitives = (str, int, bool, float)
    sequence = (tuple, list)
    for key, value in toCopy.items():
        if isinstance(value, primitives):
            copy[key] = value
        elif isinstance(value, sequence):
            copy[key] = str(value)
        elif isinstance(value, dict):
            copy[key] = dictCopy(value)
    return copy

cleanCharacteristics = dictCopy

def loadData(which):
    return (dataCharacteristics[which]['load'](),
            cleanCharacteristics(dataCharacteristics[which]))
