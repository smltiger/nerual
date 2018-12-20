import numpy as np
import gym
import multiprocessing as mp
import time
import skimage as skimage
from skimage import transform,color,exposure,io,feature

from keras import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *

import tensorflow as tf
import keras.backend.tensorflow_backend as K

import pickle
import random

class ESModel():
    def __init__(self):
        model = Sequential()
        
        model.add(Convolution2D(64,(7,7),strides=(3,3),padding='same',input_shape=(150,150,2),use_bias=False))
        model.add(Activation('tanh'))

        model.add(Convolution2D(128,(5,5),strides=(3,3),padding='same',use_bias=False))
        model.add(Activation('tanh'))

        model.add(Convolution2D(128,(4,4),strides=(2,2),padding='same',use_bias=False))
        model.add(Activation('tanh'))
        
        model.add(Convolution2D(64,(4,4),strides=(2,2),padding='same',use_bias=False))
        model.add(Activation('tanh'))

        model.add(Convolution2D(32,(3,3),strides=(1,1),padding='same',use_bias=False))
        model.add(Activation('tanh'))

        model.add(Flatten())
            
        model.add(Dense(512,use_bias=False))
        model.add(Activation('tanh'))
        model.add(Dense(3,use_bias=False))

        self.model = model
        self.weights = model.get_weights()
        self.params = self.flatten(self.weights)

        if __name__ == "__main__":
            print('**********net structure*********')
            i = 0
            for w in self.weights:
                print('layer',i,' shape:', w.shape)
                i += 1


    def flatten(self, weights):
        w = []
        for layer_weight in self.weights:
            w.append(layer_weight.reshape(layer_weight.size))
        return np.concatenate(w)
    
    def update_weights(self,params):
        split_index = 0
        w = []
        for layer_weight in self.weights:
            block_size = layer_weight.size
            block = params[split_index:split_index+block_size]
            block = block.reshape(layer_weight.shape)
            split_index += layer_weight.size
            w.append(block)

        self.model.set_weights(w)


def get_reward(params, env, human_data,seed_and_id=None, render=False):
    global model

    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        params += sign(k_id) * SIGMA * np.random.randn(params.size)
        model.update_weights(params)

    ep_r = 0.


    img_row = np.zeros((1,150,150))
    img_col = np.zeros((1,150,150))
    img = np.stack((img_row, img_col), axis=3)
    # run episode
    s = env.reset()
    s = skimage.color.rgb2gray(s)
    s = skimage.transform.resize(s,(150,150))
    s = feature.canny(s)
    s_row = np.sum(s, axis=0)
    s_col = np.sum(s, axis=1)
    done = False
    action = [3,2,1]
    info = None
    
    for step in range(MAX_STEP):
        predict = model.model.predict(img)
        action_index = np.argmax(predict)
        s_, r, done, _info = env.step(action[action_index])
        s_ = skimage.color.rgb2gray(s_)
        s_ = skimage.transform.resize(s_,(150,150))
        s_ = feature.canny(s_)
        s_row_ = np.sum(s_, axis=0)
        s_col_ = np.sum(s_, axis=1)
        col_mov = s_col_ - s_col
        row_mov = s_row_ - s_row
        col_mov = col_mov.reshape(1,1,col_mov.shape[0])
        row_mov = row_mov.reshape(1,1,row_mov.shape[0])
        img_row = np.append(col_mov, img_row[:, :149, :], axis=1)
        img_col = np.append(row_mov, img_col[:, :149, :], axis=1)
        img = np.stack((img_row, img_col), axis=3)
        
        
        if render: env.render()
        if r > 0: r = 1
        if r == 0: r = 0.01
        ep_r += r
        if info is None:    info = _info
        if _info['ale.lives'] > info['ale.lives']:
            ep_r -= 1
            #break
        if done: env.reset()
        s_col = s_col_
        s_row = s_row_
        info = _info

    return ep_r


#设置显示格式
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=160)
np.set_printoptions(formatter={'float':'{:0.3f}'.format})

GAME = 'space_invaders'
MAX_STEP = 2000
model = ESModel()
if os.path.isfile(GAME+'.h5'):
    print ("Now we load weight")
    model.model.load_weights(GAME+".h5")

env = gym.make('SpaceInvaders-v0')
sum = 0
for i in range(10):
    net_r = get_reward(model.params, env,None, None,render=True)
    print('score:', net_r)
    sum += net_r

print('average:',sum/10)
