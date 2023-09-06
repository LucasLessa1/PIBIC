import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization,Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array


class PreNeuralNetwork():

    def __init__(self, train: str, test: str, valid: str):
        self.batch_size = 32  
        self.img_height = 224 
        self.img_width  = 224
        self.channel = 3
        self.dir_train = train
        self.dir_test  = test
        self.dir_valid = valid

    def keras_data_gen(self):
        self.train_datagen= ImageDataGenerator(rescale=1/255)
        self.val_datagen  = ImageDataGenerator(rescale=1/255)
        self.test_datagen = ImageDataGenerator(rescale=1/255)
        
        self.train_generator=self.train_datagen.flow_from_directory(self.dir_train,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                class_mode='sparse',batch_size=self.batch_size)

        self.val_generator=self.val_datagen.flow_from_directory( self.dir_valid,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                class_mode='sparse',batch_size=self.batch_size)

        self.test_gemerator=self.test_datagen.flow_from_directory(self.test_datagen,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                class_mode='sparse',batch_size=self.batch_size)
        
    
    
class ResNet101(PreNeuralNetwork):

    def __init__(self):
        super().__init__(self)
        
    
    def nn_config(self):
        from keras.applications import ResNet101V2
        convlayer = ResNet101V2(input_shape=(self.img_height, self.img_width, self.channel), weights='imagenet',include_top=False)
        for layer in convlayer.layers:
            layer.trainable=False