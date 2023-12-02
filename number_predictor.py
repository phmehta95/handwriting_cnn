import numpy as np
import tensorflow as tf
import keras.utils as image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
import sys
import time
from PIL import Image 
import cv2

reconstructed_model = tf.keras.models.load_model("/home/pruthvi/Desktop/handwriting_cnn/handwriting_cnn.h5")

directory = "/home/pruthvi/Desktop/handwriting_cnn/external_test_data/number6.png"

#Predicting images
image = cv2.imread(directory, cv2.IMREAD_GRAYSCALE) #convert to greyscale 
image = cv2.resize(image, (28,28), interpolation=cv2.INTER_LINEAR)
image = cv2.bitwise_not(image)
np.set_printoptions(suppress=True,linewidth=np.nan)
#image = np.asarray(image)
#image = image/255

print(image)
print()
print(image.shape)

plt.imshow(image)
plt.show()
image = image.reshape(784)

print(image.shape)


pred = reconstructed_model.predict(np.expand_dims(image,0))
print(pred.argmax())

#classes = reconstructed_model.predict(img)
#print(classes)
####can't do it like this, need to convert image of number to MNIST -> return pixel value of each pixel in the image#####
