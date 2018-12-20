
import numpy as np
import skimage as skimage
from skimage import transform,color,exposure,feature
import gym
import pickle


env = gym.make('PongNoFrameskip-v0')
env.reset()
record = False
D = []

from pynput import keyboard

def store_step(s, reward):
    s = skimage.color.rgb2gray(s)
    s = s[35:175,:]
    s = feature.canny(s)
    s = s.reshape(1,s.shape[0],s.shape[1],1)
    D.append([s,reward])

def save_step():
    global D
    with open('data.pickle', 'wb') as f:
        pickle.dump(D, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def on_press(key):
    global env,record
    try:
        if key.char == 'j':
            s,r,done,info=env.step(5)
            env.render()
            if record:
                store_step(s,[0,0,0.01])
        elif key.char == 'k':
            s,r,done,info=env.step(4)
            env.render()
            if record:
                store_step(s,[0,0.01,0])
        elif key.char == 'a':
            s,r,done,info=env.step(0)
            env.render()
            if record:
                store_step(s,[0.01,0,0])
        elif key.char == 'z':
            record = not record
            print('record:',record)
        elif key.char == 's':
            save_step()        
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
