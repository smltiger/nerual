import numpy as np

import gym

import os

import gc

import random

from keras import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *

import skimage as skimage
from skimage import transform,color,exposure,io,feature

from collections import deque

from time import sleep
  


def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), subsample=(4, 4), border_mode='same',input_shape=(80,80,2)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(3))
   
    adam = Adam(lr=1e-4)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

action = [3,2,1]

env = gym.make('SpaceInvaders-v0')
img_row = np.zeros((1,80,80))
img_col = np.zeros((1,80,80))
img = np.stack((img_row, img_col), axis=3)
# run episode
s = env.reset()
s = skimage.color.rgb2gray(s)
s = skimage.transform.resize(s,(80,80))
s = feature.canny(s)
s_row = np.sum(s, axis=0)
s_col = np.sum(s, axis=1)

D = deque()

model = buildmodel()

if os.path.isfile('model.h5'):
    print('load model...')
    model.load_weights('model.h5')
    model.compile(loss='mse', optimizer = Adam(lr=args.LEARNING_RATE))
    print('model loaded.')

OBSERVATION = 100
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
EXPLORE = 100000
REPLAY_MEMORY = 5000
BATCH = 32
GAMMA = 0.99

t = 0
epsilon = 0.2
OBSERVE = OBSERVATION
step_reward = 0.0
while(True):
    loss = 0
    Q = 0
    action_index = 0
    r = 0
    a = action[action_index]

    if random.random() <= epsilon:
        print('-----------------Random Action-----------')
        action_index = random.randrange(3)
        a = action[action_index]
    else:
        Q = model.predict(img)
        print(Q)
        action_index = np.argmax(Q)
        a = action[action_index]

    if epsilon > FINAL_EPSILON and t > OBSERVE:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    s_,r,done,info_ = env.step(a)
    env.render()
    s_ = skimage.color.rgb2gray(s_)
    s_ = skimage.transform.resize(s_,(80,80))
    s_ = feature.canny(s_)
    s_row_ = np.sum(s_, axis=0)
    s_col_ = np.sum(s_, axis=1)
    col_mov = s_col_ - s_col
    row_mov = s_row_ - s_row
    col_mov = col_mov.reshape(1,1,col_mov.shape[0])
    row_mov = row_mov.reshape(1,1,row_mov.shape[0])
    img_row = np.append(col_mov, img_row[:, :79, :], axis=1)
    img_col = np.append(row_mov, img_col[:, :79, :], axis=1)
    img_ = np.stack((img_row, img_col), axis=3)

    if r > 0: r = 1
    D.append([img,action_index,r,img_,done])
    if len(D) > REPLAY_MEMORY:
        tmp = D.popleft()
        del tmp
    if done:
        env.reset()
    if t > OBSERVE:
        minibatch = random.sample(D, BATCH)
        s_t, a_, r_t, s_t1, done = zip(*minibatch)
        s_t = np.concatenate(s_t)
        s_t1 = np.concatenate(s_t1)
        targets = model.predict(s_t)
        Q_ = model.predict(s_t1)
        targets[range(BATCH),a_] = r_t + GAMMA*np.max(Q_, axis=1)*np.invert(done)
        
        loss += model.train_on_batch(s_t, targets)

    s_col = s_col_
    s_row = s_row_
    img = img_
    info = info_
    t = t + 1

    if t%1000 == 0:
        print('saving model...')
        model.save_weights('model.h5', overwrite=True)
        gc.collect()

    print('Step:',t,' action:',action_index, ' reward:',r, ' loss:',loss)


    
        

