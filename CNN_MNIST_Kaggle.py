#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:20:42 2018

@author: susopeiz
"""

#This program identifies the handwritten digits from the MNIST dataset using a
#Convolutional Neural Network


#Preprocessing data in csv format to (nx,ny) shape for the CNN
def preprocessing_MNISTdata():
    
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    os.chdir('/Volumes/Suso/MachineLearning/Kaggle/mnist/')
    
    nx=28
    ny=28
    
    
    #Train data
    data_train=pd.read_csv('train.csv')
    labels_train=data_train.label.values
    images_train=data_train.iloc[:,1:].values/255
    images_train=images_train.reshape(len(data_train),nx,ny)
    images_train=np.expand_dims(images_train,axis=3)
    
    from keras.utils.np_utils import to_categorical
    
    #labels in categorical format
    labels10_train=to_categorical(labels_train, num_classes=10)
            
    
    #Test data
    data_test=pd.read_csv('test.csv')
    images_test=data_test.values/255
    images_test=images_test.reshape(len(data_test),nx,nx)
    images_test=np.expand_dims(images_test,axis=3)

    return images_train,images_test,labels10_train



#Construction of the CNN with several convolutional layers plus several fully 
#connected layers
def constructing_CNN():

    
    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
    from keras.layers import Dropout
    import numpy as np
    
    
    
    #Initializing the CNN
    classifier=Sequential()
    
    #Adding layers. 
    
    #Convolutions
    ##ORDER OF THE INPUT_SHAPE IS NX,NY,Nchannels
    classifier.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
    
    #Pooling
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    #Second CNV layer
    classifier.add(Conv2D(64,(3,3),activation='relu'))
    
    #Pooling
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    ##Third CNV layer
    #classifier.add(Conv2D(32,(3,3),activation='relu'))
    #
    ##Pooling
    #classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    #Flatten
    classifier.add(Flatten())
    
    #Fully connected layers
    #First layer
    classifier.add(Dense(units=128,activation='relu'))
    
    #Dropout for avoiding overfitting
    classifier.add(Dropout(rate=0.5))
    
    #First layer
    classifier.add(Dense(units=64,activation='relu'))
    #Second layer
    classifier.add(Dense(units=10,activation='softmax'))
    
    #compiling the ANN
    classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


    return classifier



images_train,images_test,labels10_train=preprocessing_MNISTdata()
CNN=constructing_CNN()


#Training!  
CNN.fit(images_train, labels10_train, batch_size = 1024, epochs = 30,validation_split=0.1)


#Predicting test set in categorical format (probabilities)
labels10_test=CNN.predict(images_test)

#Maximum probability gives the forecasted label in decimal format
labels_test=np.argmax(labels10_test,axis=1)

#Forecasted labels to a df and printed in a csv file
df_test=pd.DataFrame(data=np.hstack((np.arange(1,28001).reshape(-1,1),labels_test.reshape(-1,1))),columns=['ImageId','Label'])
df_test.to_csv('Prediction_CNN.csv',index=False)
