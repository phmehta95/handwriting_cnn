#Loading MNIST dataset
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from numpy import load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from keras.callbacks import CSVLogger

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


#Defining baseline model
#The Dense layer is the layer that contains all the neurons that are deeply connected within themselves -  every neuron in the dense layer takes the input from all the other neurons of the previous layer.

#Relu and Softmax are types of activation functions - see the README for more information about activation functions

#Categorical_crossentropy is a loss function that can be used when there are two or more label classes. If we were using integers as labels instead of one_hot encoding we would use the Keras SparseCategoricalCrossentropy loss models.

#ADAM is a gradient descent algorithm used to calculate the weights for the model - see the README for more information about how NN weights are calculated.

def baseline_model():
    #Create model
    model = Sequential()
    model.add(Dense(num_pixels, input_shape=(num_pixels,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#Building the baseline model
model = baseline_model()

#Setting up csv_logger to save model history

csv_logger = CSVLogger("training.log", separator=",", append="False")

#Fit the model
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=200, verbose=2, callbacks=[csv_logger])

# Final evaluation of the model
scores = model.evaluate(testX, testY, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

#Saving model to .h5 file for later use
model.save("handwriting_cnn.h5")




