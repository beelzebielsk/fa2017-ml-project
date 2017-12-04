import numpy as np

def loadImages():
    path = "/home/adam/Documents/scratch/faces/face_images.npz"
    faceStuff = np.load(path)
    # Of shape (imageHeight, imageWidth, images)
    data = faceStuff['face_images']
    height, weight, images = data.shape

    # I want to reshape to (images, imageHeight*imageWidth)
    data = np.rollaxis(data, 2, 0)
    return data.reshape(images, height*weight)
