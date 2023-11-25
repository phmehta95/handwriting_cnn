#Loading MNIST dataset
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from numpy import load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

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

#Flatten 28*28 images to a 794 length pixel vector for each image

num_pixels = trainX.shape[1] * trainX.shape[2]
print (num_pixels)

#Reducing memory requirements to be 32-bit (default precision type used by Keras)
trainX = trainX.reshape((trainX.shape[0], num_pixels)).astype('float32')
testX = testX.reshape((testX.shape[0], num_pixels)).astype('float32')

#Scaling grayscale value inputs so instead of being between 0 and 255, they are between 0 and 1
trainX = trainX/255
testX = testX/255

print (trainX[1],testX[2])

#Multi-class classification and one-hot encoding
#One-hot encoding is used to represent categorical variables as numerical values in a machine learning model - this avoids problem in label-encdoing where labels assigned to the categorical variables. For our classifier, the output values are numbers from 0-9 which if each output number has a numerical label applied to it would produce an ordinal relationship between them -> this could lead to a bias in the weights applied by a NN  - so using one-hot encoding would remove this ordinal relationship. This is done by using the "to_categorical" function in Keras.

#One hot encoder outputs

trainY = to_categorical(trainY)
testY = to_categorical(testY)
num_classes = testY.shape[1]
