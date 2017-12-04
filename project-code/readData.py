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

    def nextBatch(self, amount):
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
        return self.images[batchImages, :]

class faces:
    def __init__(self, images, testRatio=.5):
        numTest = int(images.shape[0] * testRatio)
        self.test = batch(images[:numTest])
        self.train = batch(images[numTest:])
