
import numpy as np
import skimage as skimage
from skimage import transform,color,exposure,feature
import gym
import pickle


env = gym.make('MontezumaRevenge-v0')
env.reset()

record = False
D = []
reward = 0.1
print(env.action_space)
print(env.observation_space)
print('record:', record)
print('now reward:',reward)

from pynput import keyboard

def store_step(a):
    D.append(a)

def save_step():
    with open('Montezuma.pickle', 'wb') as f:
        pickle.dump(D, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def on_press(key):
    global env,record,reward
    try:
        if key.char == 'w':
            a = 2
            s,r,done,info=env.step(a)
            env.render()
            if record:
                store_step(a)
        elif key.char == 's':
            a = 5
            s,r,done,info=env.step(a)
            env.render()
            if record:
                store_step(a)
        elif key.char == 'a':
            a = 4
            s,r,done,info=env.step(a)
            env.render()
            if record:
                store_step(a)
        elif key.char == 'd':
            a = 3
            s,r,done,info=env.step(a)
            env.render()
            if record:
                store_step(a)
        elif key.char == 'j':
            a = 12
            s,r,done,info=env.step(a)
            env.render()
            if record:
                store_step(a)
        elif key.char == 'k':
            a = 11
            s,r,done,info=env.step(a)
            env.render()
            if record:
                store_step(a)
        elif key.char == 'i':
            a = 1
            s,r,done,info=env.step(a)
            env.render()
            if record:
                store_step(a)
        elif key.char == 'o':
            a = 0
            s,r,done,info=env.step(a)
            env.render()
            if record:
                store_step(a)
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
        elif key.char == '1':
            state = env.env.clone_full_state()
            with open('1.pickle','wb') as state_file:
                pickle.dump(state, state_file)
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
