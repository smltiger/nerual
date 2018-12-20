from threading import Thread

from time import sleep

from pynput import keyboard

import retro

import numpy as np

from collections import deque

import skimage as skimage
from skimage import transform,color,exposure

import pickle

from net import buildmodel

import os

from keras import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *

import random

action_index = 0
D = deque()

def tran_img(img):
    img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img, (80,80))
    img = skimage.exposure.rescale_intensity(img, out_range=(0,255))
    img = img / 255.0
    return img 
    
def on_press(key):
    global action_index
    if key.char == 'w':
        action_index = 1
    elif key.char == 's':
        action_index = 2
    elif key.char == 'a':
        action_index = 3
    elif key.char == 'd':
        action_index = 4
    elif key.char == 'j':
        action_index = 0
        
def on_release(key):
    global action
    if key.char == 'j':
        action_index = 0
    else:
        #action_index = 0
        pass
    
def async(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target = f, args = args, kwargs = kwargs)
        thr.start()
    return wrapper

@async
def Thread1():
    with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
        listener.join()
    

def Thread2():
    global action_index,D
    BATCH = 32
    REPLAY_MEMORY = 10000
    
    model = buildmodel()
    if os.path.isfile('model.h5'):
        print('load model...')
        model.load_weights('model.h5')
        model.compile(loss='mse', optimizer = Adam(lr=1e-4))
        print('model loaded.')

    env = gym.make(game='Pong-v0')
    env.reset()
    s,r,done,info=env.step(random.randint(0,5))
    s = tran_img(s)
    s = np.stack((s,s,s,s),axis=2)
    s = s.reshape(1, s.shape[0], s.shape[1], s.shape[2])

    t = 0
    while True:
        s_,r,done,info=env.step(action_index)
        env.render()
        s_ = tran_img(s_)
        s_ = s_.reshape(1,s_.shape[0],s_.shape[1],1)
        s_ = np.append(s_, s[:,:,:,:3], axis=3)
        r = r/50 + 0.1

        D.append((s, action_index, r, s_, done))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        
        if len(D) >= REPLAY_MEMORY and t % 5000 == 0:
            print('step:',t,'traing agent...')

            for i in range(5000):
                minibatch = random.sample(D, BATCH)
                s_t, a_t, r_t, s_t1, done_ = zip(*minibatch)
                s_t = np.concatenate(s_t)
                s_t1 = np.concatenate(s_t1)
                targets = model.predict(s_t)
                Q_ = model.predict(s_t1)
                targets[range(BATCH),a_t] = r_t + 0.99*np.max(Q_, axis=1)*np.invert(done_)
                model.train_on_batch(s_t, targets)

            print('saving trained agent...')
            model.save_weights('model.h5',overwrite=True)
            
        if done:
            print('game over...restart a new game...')
            env.reset()
        sleep(0.01)

        t += 1
        

Thread1()
Thread2()
