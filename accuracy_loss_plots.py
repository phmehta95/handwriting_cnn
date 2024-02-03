from tensorflow.keras.datasets import mnist
import tensorflow as tf
import pandas as pd
from keras.callbacks import CSVLogger
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import matplotlib as plt
import dataframe_image as dfi
from yellowbrick.classifier import ClassificationReport
from sklearn.neighbors import KNeighborsClassifier


#Plotting the Training and Testing accuracy                                     

#CNN model                                                             
reconstructed_model = tf.keras.models.load_model("/home/pruthvi/Desktop/handwriting_cnn/handwriting_cnn.h5")

#Training log for original model                                                
log_data = pd.read_csv("training.log", sep=',' , engine='python')

#Plotting the Training and Testing accuracy                                             
import matplotlib.pyplot as plt
acc = log_data['accuracy']
val_acc = log_data['val_accuracy']
loss = log_data['loss']
val_loss = log_data['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'm', label='Validation loss')

plt.title('Training and validation accuracy and loss')
plt.legend(loc=0)

#Function to plot a confusion matrix                                            

import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

def evaluate(reconstructed_model):
    (trainX, trainY),(testX,testY) = mnist.load_data()
    
    dataGen = ImageDataGenerator(rotation_range=15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,zoom_range=[0.5,2],validation_split=0.2)

    trainX = trainX.reshape(60000,28,28,1)
    testX = testX.reshape((testX.shape[0], 28 * 28 * 1))
    trainY = trainY.reshape(60000,1,1,1)
    
    train_generator = dataGen.flow(trainX, trainY, shuffle=True, seed=2, save_to_dir=None, subset='training')

    validation_generator = dataGen.flow(trainX, trainY, shuffle=True, seed=2, save_to_dir=None, subset='validation')

    batch_size = 100
    #num_test_samples = len(validation_generator.filenames)
    num_test_samples = 10000
    
    #Y_pred = reconstructed_model.predict_generator(validation_generator, num_test_samples // batch_size + 1)
    Y_pred = reconstructed_model.predict(testX)
    Y_pred = Y_pred.argmax(axis=1)

    print(confusion_matrix(testY, Y_pred))
    global cfm
    cfm = confusion_matrix(testY, Y_pred)

    #Making Classification Report

    print('\n\nClassification Report\n')
    target_names = ["0","1","2","3","4","5","6","7","8","9"]
    print(classification_report(testY, Y_pred, target_names\
=target_names))
    global crp
    crp = classification_report(testY, Y_pred, target_names\
=target_names, output_dict=True)
    
evaluate(reconstructed_model)
print(cfm)


#Making Confusion Matrix look pretty

classes = ["0","1","2","3","4","5","6","7","8","9"]

df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True, cmap = "BuPu", fmt="d")
plt.xlabel("Classifier Prediction")
plt.ylabel("Actual [Truth]")
cfm_plot.figure.savefig("confusion_matrix.png")


#Plotting classification report
df_crp = pd.DataFrame(crp)
print(df_crp.to_string())
crp_plot = df_crp.iloc[:3, :10].T.plot(kind='bar',figsize=(12,7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
crp_plot.figure.savefig("classification_report.png")
plt.show()
