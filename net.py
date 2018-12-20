from keras import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *

import numpy as np


class Args():
    def __init__(self):
        self.img_rows = 150
        self.img_cols = 150
        self.img_channels = 2
        self.LEARNING_RATE = 1e-4
        
def buildmodel():
    print('now we build model...')
    args = Args()
    model = Sequential()
    model.add(Convolution2D(128,(8,8),strides=(4,4),padding='same',input_shape=(args.img_rows,args.img_cols,args.img_channels)))
    model.add(Activation('relu'))
    model.add(Convolution2D(512,(7,7),strides=(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512,(4,4),strides=(2,2),padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512,(3,3),strides=(2,2),padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(3))

    adam = Adam(lr=args.LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print('we finish building the model')

    return model

