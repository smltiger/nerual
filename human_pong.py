
import numpy as np
import skimage as skimage
from skimage import transform,color,exposure,feature
import gym
import pickle


env = gym.make('PongNoFrameskip-v0')
env.reset()

record = False
D = []
reward = 0.1
print(env.action_space)
print(env.observation_space)
print('record:', record)
print('now reward:',reward)

from pynput import keyboard

def store_step(s, reward):
    s = skimage.color.rgb2gray(s)
    s = feature.canny(s)
    s = s.reshape(1,s.shape[0],s.shape[1],1)
    D.append([s,reward])

def save_step():
    with open('Pong.pickle', 'wb') as f:
        pickle.dump(D, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def on_press(key):
    global env,record,reward
    try:
        if key.char == 'w':
            s,r,done,info=env.step(2)
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
            s,r,done,info=env.step(4)
            env.render()
            if record:
                a = np.zeros(8)
                a[4] = reward
                store_step(s,a)
        elif key.char == 'd':
            s,r,done,info=env.step(3)
            env.render()
            if record:
                a = np.zeros(8)
                a[3] = reward
                store_step(s,a)
        elif key.char == 'j':
            s,r,done,info=env.step(12)
            env.render()
            if record:
                a = np.zeros(8)
                a[6] = reward
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
            reward += 0.2
            print('now reward is:',reward)
        elif key.char == '-':
            reward -= 0.2
            print('now reward is:',reward)
        elif key.char == 'r':
            print('resetting env...')
            env.reset()
            env.step(0)
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
