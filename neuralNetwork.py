import os                
import zipfile                
import shutil            
import glob         

#For math operations           
import time               
import random               

#For images operations
import cv2             
import pydicom          
import scipy
from PIL import Image     
from numpy import expand_dims  
from google.protobuf import builder as _builder


import keras     
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization,Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.optimizers import Adam



class PreNeuralNetwork():
    def __init__(self):
        pass
    

    def create_metrics_dataframe(self, history):
        
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(history.history['loss']) + 1),
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'train_accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy']
        })
        return metrics_df


    def evaluate(self, test_data):
        return self.model.evaluate(test_data)


    def plot_history(self, history):
        '''
        This role belongs to Matheus Vieira, Project Director (until 2022) at IEEE-CIS. 
        Not only did he help us with this project, he also contributed a lot of ideas. 
        We leave here our thanks to him.
        
        The purpose of this function is to trace the learning line of the neural network, 
        plotting the training and validation loss.
        '''
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
        




class ResNet101Model(PreNeuralNetwork):
    def __init__(self, num_classes, image_shape):
        # super().__init__(train='', test='', valid='')  # Adjust directories accordingly
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])

    def build_model(self):
        base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=self.image_shape)
        
        output = base_model.output
        output = GlobalAveragePooling2D()(base_model.output)
        output = Dense(1024, activation='relu')(output)
        output = BatchNormalization()(output)
        
            
        predictions = Dense(self.num_classes, activation='softmax')(output)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        
       
        return self.model
    
    def train(self, train_data, val_data, model,  epochs, batch_size, early_stop_patience):
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop_callback]
        )
        
        return self.history



class EfficientNetModelB7(PreNeuralNetwork):
    def __init__(self, num_classes, image_shape):
        super().__init__()  # Adjust directories accordingly
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])

    def build_model(self):
        base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=self.image_shape)
        
        output = base_model.output
        output = GlobalAveragePooling2D()(base_model.output)
        output = Dense(1024, activation='relu')(output)
        output = BatchNormalization()(output)
        
            
        predictions = Dense(self.num_classes, activation='softmax')(output)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        
       
        return self.model
    
    def train(self, train_data, val_data, model,  epochs, batch_size, early_stop_patience):
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop_callback]
        )
        
        return self.history



class EfficientNetModelB4(PreNeuralNetwork):
    def __init__(self, num_classes, image_shape):
        # super().__init__(train='', test='', valid='')  # Adjust directories accordingly
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])

    def build_model(self):
        base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=self.image_shape)
        
        output = base_model.output
        output = GlobalAveragePooling2D()(base_model.output)
        output = Dense(1024, activation='relu')(output)
        output = BatchNormalization()(output)
        
            
        predictions = Dense(self.num_classes, activation='softmax')(output)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        
       
        return self.model
    
    def train(self, train_data, val_data, model,  epochs, batch_size, early_stop_patience):
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop_callback]
        )
        
        return self.history
    



print("Working Well")
