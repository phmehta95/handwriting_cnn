import numpy as np
import tensorflow as tf
import keras.utils as image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
import sys
import time


reconstructed_model = tf.keras.models.load_model("/home/pruthvi/Desktop/handwriting_cnn/handwriting_cnn.h5")

directory = "/home/pruthvi/Desktop/handwriting_cnn/external_test_data/number1.png"

#Predicting images

img = image.load_img("/home/pruthvi/Desktop/handwriting_cnn/external_test_data/number1.png", target_size=(28,28))

plt.imshow(img)
plt.show()

classes = reconstructed_model.predict(img)
print(classes)
####can't do it like this, need to convert image of number to MNIST -> return pixel value of each pixel in the image#####
