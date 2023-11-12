#Loading MNIST dataset
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from numpy import load

(trainX, trainY),(testX,testY) = mnist.load_data()


#Output print statement summarising the dataset
#trainX is the training data set
#trainY is the set of labels given to all the data in trainX

print("TrainingX:", trainX.shape)
print("TrainingY:", trainY.shape)
print("TestingX:", testX.shape)
print("TestingY:", testY.shape)


#For the first number in the training  set, we can output the shape 
print("TrainingX:", trainX[0].shape)
print("TrainingY:", trainY[0].shape)

#We can also output the type - we can see for trainX these are arrays as these are the 28x28 pixel images. For trainY the type is just an integer, as this is just a label assigned to the image 
print(type(trainX[0]))
print(type(trainY[0]))

#You can also visualise the 28*28 images on the screen by just printing a training image in the array of 60000 images. Note how the 28*28 pixel array has a value assigned to each point - that is the grayscale value of each pixel i.e. how dark the pixel is. The max value is 255, min is 0
#print(trainX[4])

#The associated label 0-9 can also be printed for each image of a number
#print(trainY[4])

# plot first five images

for i in range(5):
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
    plt.pause(1)
    plt.show(block=False)
    plt.close()
