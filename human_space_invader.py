
import numpy as np
import skimage as skimage
from skimage import transform,color,exposure,feature
import gym
import pickle
import time
import random


env = gym.make('SpaceInvaders-v0')
s_t0 = env.reset()
img_row = np.zeros((1,150,150))
img_col = np.zeros((1,150,150))
img = np.stack((img_row, img_col), axis=3)
s = skimage.color.rgb2gray(s_t0)
s = skimage.transform.resize(s,(150,150))
s = feature.canny(s)
s_row = np.sum(s, axis=0)
s_col = np.sum(s, axis=1)

record = False
D = []
reward = 0.002
print(env.action_space)
print(env.observation_space)
print('record:', record)
print('now reward:',reward)

from pynput import keyboard

def store_step(s_, reward):
    global img_row,img_col,img,s,s_row,s_col
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
    D.append([img,reward])

    s_col = s_col_
    s_row = s_row_
    

def save_step():
    with open('space_invaders.pickle', 'wb') as f:
        pickle.dump(D, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def on_press(key):
    global env,record,reward
    try:
        if key.char == 'w':
            s,r,done,info=env.step(3)
            env.render()
            if record:
                a = np.zeros(8)
                a[2] = reward
                store_step(s,a)
        elif key.char == 's':
            s,r,done,info=env.step(5)
            env.render()
            if record:
                a = np.zeros(8)
                a[5] = reward
                store_step(s,a)
        elif key.char == 'a':
            s,r,done,info=env.step(3)
            env.render()
            if record:
                a = np.zeros(3)
                a[0] = reward
                store_step(s,a)
        elif key.char == 'd':
            s,r,done,info=env.step(2)
            env.render()
            if record:
                a = np.zeros(3)
                a[1] = reward
                store_step(s,a)
        elif key.char == 'j':
            s,r,done,info=env.step(1)
            env.render()
            if record:
                a = np.zeros(3)
                a[2] = reward
                store_step(s,a)
        elif key.char == 'k':
            s,r,done,info=env.step(11)
            env.render()
            if record:
                a = np.zeros(8)
                a[7] = reward
                store_step(s,a)
        elif key.char == 'i':
            s,r,done,info=env.step(1)
            env.render()
            if record:
                a = np.zeros(8)
                a[1] = reward
                store_step(s,a)
        elif key.char == 'o':
            s,r,done,info=env.step(0)
            env.render()
            if record:
                a = np.zeros(8)
                a[0] = reward
                store_step(s,a)
        elif key.char == 'l':
            s,r,done,info=env.step(0)
            env.render()
        elif key.char == 'z':
            record = not record
            print('record:',record)
        elif key.char == 'x':
            save_step()        
        elif key.char == '+':
            reward += 0.01
            print('now reward is:',reward)
        elif key.char == '-':
            reward -= 0.01
            print('now reward is:',reward)
        elif key.char == '*':
            reward *= 2
            print('now reward is:',reward)
        elif key.char == '/':
            reward /= 2
            print('now reward is:',reward)
        elif key.char == 'r':
            print('resetting env...')
            env.reset()
            s,r,done,info = env.step(0)
            print(info)
            for i in range(2000):
                env.render()
    except AttributeError:
        print('special key {0} pressed'.format(key))

def on_release(key):
    global env
    print('{0} released'.format(key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
