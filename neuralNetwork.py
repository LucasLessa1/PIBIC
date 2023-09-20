# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization,Activation
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import load_img,img_to_array
from imports_ import *

# class PreNeuralNetwork():

#     def __init__(self, train: str, test: str, valid: str):
#         self.batch_size = 32  
#         self.img_height = 224 
#         self.img_width  = 224
#         self.channel = 3
#         self.dir_train = train
#         self.dir_test  = test
#         self.dir_valid = valid

#     def keras_data_gen(self):
#         self.train_datagen= ImageDataGenerator(rescale=1/255)
#         self.val_datagen  = ImageDataGenerator(rescale=1/255)
#         self.test_datagen = ImageDataGenerator(rescale=1/255)
        
#         self.train_generator=self.train_datagen.flow_from_directory(self.dir_train,
#                                                 target_size=(224,224),
#                                                 color_mode='rgb',
#                                                 class_mode='sparse',batch_size=self.batch_size)

#         self.val_generator=self.val_datagen.flow_from_directory( self.dir_valid,
#                                                 target_size=(224,224),
#                                                 color_mode='rgb',
#                                                 class_mode='sparse',batch_size=self.batch_size)

#         self.test_gemerator=self.test_datagen.flow_from_directory(self.test_datagen,
#                                                 target_size=(224,224),
#                                                 color_mode='rgb',
#                                                 class_mode='sparse',batch_size=self.batch_size)
        
    
    

class ResNet101Model:
    def __init__(self, num_classes, image_shape):
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.model = self.build_model()
        self.metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
    
    def build_model(self):
        base_model = ResNet101(weights='imagenet', include_top=False, input_shape=self.image_shape)
        for layer in base_model.layers:
            layer.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, train_data, val_data, epochs, batch_size, early_stop_patience=5):
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop_callback]
        )
        
        return self.history
    
    def update_metrics(self, history):
        for epoch, train_loss, val_loss, train_accuracy, val_accuracy in zip(
                range(len(history.history['loss'])),
                history.history['loss'],
                history.history['val_loss'],
                history.history['accuracy'],
                history.history['val_accuracy']
        ):
            self.metrics_df = self.metrics_df.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy
            }, ignore_index=True)
        return self.metrics_df
    def evaluate(self, test_data):
        return self.model.evaluate(test_data)



print("Working Well")