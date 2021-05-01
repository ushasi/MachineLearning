# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:49:49 2019
@author: ushasi2
help:
https://keras.io/applications/#resnet
"""
import matplotlib.pyplot as plt
#import tensorflow
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import vgg16
from tensorflow.python.keras.models import Sequential, Model
#from keras.preprocessing import image
from tensorflow.keras.layers import Dense,  Flatten#Activation,
from tensorflow.python.keras.layers import Dense, Dropout
import scipy.io as sio
import numpy as np
import os


NUM_CLASSES = 125		####
CHANNELS = 3
IMAGE_RESIZE = 224	####
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 8
#STEPS_PER_EPOCH_TRAINING = 10
#STEPS_PER_EPOCH_VALIDATION = 10
BATCH_SIZE_TRAINING = 64	####
BATCH_SIZE_VALIDATION = 64	####
BATCH_SIZE_TESTING = 64




image_size = 224
os.environ["CUDA_VISIBLE_DEVICES"]="1"

vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze all the layers
for layer in vgg_conv.layers[:]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)


# Create the model
model = Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(125, activation='softmax'))

#model.layers[0].trainable = True
# Show a summary of the model. Check the number of trainable parameters
model.summary()

# Load the normalized images
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 100
val_batchsize = 10
test_batchsize = 12500

train_dir = 'sketch/tx_000000000000'
validation_dir = 'sketch/tx_000000000000'
test_dir = 'sketch/tx_000000000000'
# Data generator for training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')


# Data generator for validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Configure the model for training
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the model
history = model.fit(
      train_generator,
      steps_per_epoch=
         train_generator.samples/train_generator.batch_size,
      epochs=30,
      validation_data=validation_generator, 
      validation_steps=
         validation_generator.samples/validation_generator.batch_size,
      verbose=1)



# Get the predictions from the model using the generator
predictions = model.predict(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
'''
# Run the function to illustrate accuracy and loss
visualize_results(history)
# Run the function to get the list of classes and errors
idx2label, errors, fnames = obtain_errors(validation_generator, predictions)
# Run the function to illustrate the error cases
show_errors(idx2label, errors, predictions, fnames)
'''
predictions = model.predict(validation_generator,verbose=1)



# NOTE that flow_from_directory treats each sub-folder as a class which works fine for training data
# Actually class_mode=None is a kind of workaround for test data which too must be kept in a subfolder

# batch_size can be 1 or any factor of test dataset size to ensure that test dataset is samples just once, i.e., no data is left out

test_generator = test_datagen.flow_from_directory(
    directory = 'photo/tx_000000000000',
    target_size = (image_size, image_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
)

#Need to compile layer[0] for extracting the 256- dim features.
#model.layers[1].compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

test_generator.reset()
#pred = model.layers[1].predict_generator(test_generator, steps = len(test_generator), verbose = 1) 
f =  Model(inputs=model.input,outputs=model.get_layer('dense').output)
f.compile(optimizer = optimizers.RMSprop(lr=1e-4), loss ='categorical_crossentropy', metrics = ['acc']) 
for idx,layer in enumerate(f.layers):
        if(layer.name == 'dense'):
                print(idx,layer.name)
                #print(layer.get_weights())
                print("________________")

pred = f.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

sio.savemat('vggsketch.mat',mdict={'feature':pred})

print(pred.shape)

