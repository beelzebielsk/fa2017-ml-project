import numpy as np

def loadImages():
    faceStuff = np.load("/home/adam/Documents/scratch/faces/face_images.npz")
    # Of shape (imageHeight, imageWidth, images)
    data = faceStuff['face_images']
    height, weight, images = data.shape

    # I want to reshape to (images, imageHeight*imageWidth)
    data = np.rollaxis(data, 2, 0)
    return data.reshape(images, height*weight)
