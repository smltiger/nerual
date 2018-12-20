"""
Simple code for Distributed ES proposed by OpenAI.
Based on this paper: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
Details can be found in : https://arxiv.org/abs/1703.03864

Visit more on my tutorial site: https://morvanzhou.github.io/tutorials/
"""
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


def sign(k_id): return -1. if k_id % 2 == 0 else 1.



class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v

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
    
    for step in range(MAX_STEP):
        predict = model.model.predict(img)
        action_index = np.argmax(predict)
        s_, r, done, info = env.step(action[action_index])
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
        if info['ale.lives'] < 3:
            ep_r -= 1
            break

        s_col = s_col_
        s_row = s_row_

    minibatch = random.sample(human_data, BATCH)
    state,reward = zip(*minibatch)
    state = np.concatenate(state)
    target = model.model.predict(state)
    tmp = np.argmax(target,axis=1)
    from keras.utils import to_categorical
    tmp = to_categorical(tmp,num_classes=3)
    ep_r += np.sum(tmp*reward)

    return ep_r




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


def train(mar_kid, params, env, human_data, utility, pool):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)    # mirrored sampling
    
    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, (params, env,human_data,
                                          [noise_seed[k_id], k_id],False)) for k_id in range(N_KID*2)]
    
    rewards = np.array([j.get() for j in jobs])
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward
    print(rewards)

    cumulative_update = np.zeros_like(params)       # initialize update values
    for ui, k_id in enumerate(kids_rank):
        np.random.seed(noise_seed[k_id])                # reconstruct noise using seed
        cumulative_update += utility[ui] * sign(k_id) * np.random.randn(params.size)

    LR = 0.03 if mar_kid is None else 0.03 * rewards.mean() / mar_kid
    #print('mar_kid:',mar_kid,' kid_reward.mean()',rewards.mean(),' LR:',LR)
    optimizer = SGD(model.params, LR)
    gradients = optimizer.get_gradients(cumulative_update/(2*N_KID*SIGMA))
    return gradients, rewards

#设置显示格式
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=160)
np.set_printoptions(formatter={'float':'{:0.3f}'.format})

#禁止完全分配GPU内存，设置为按需分配
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


#设置超参数
N_KID = 15                  # half of the training population
N_GENERATION = 5000000         # training step
SIGMA = 0.03                 # mutation strength or step size
N_CORE = mp.cpu_count()-1
MAX_STEP = 3000
GAME = 'space_invaders'
BATCH = 512

model = ESModel()
if os.path.isfile(GAME+'.h5'):
    print ("Now we load weight")
    model.model.load_weights(GAME+".h5")

#主程序
if __name__ == "__main__":
    # utility instead reward for update parameters (rank transformation)
    base = N_KID * 2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base
    # training
    env = gym.make('SpaceInvaders-v0')
    with open(GAME+'.pickle', 'rb') as f:
        human_data = pickle.load(f)
        f.close()
        print('human data length:',len(human_data))

    pool = mp.Pool(processes=N_CORE)
    mar = None      # moving average reward
    mar_kid = None
    for g in range(N_GENERATION):
        print('*******Gen:',g)
        t0 = time.time()
        
        gradients,kid_rewards = train(mar_kid, model.params,env,human_data, utility, pool)
        model.params += gradients
        model.update_weights(model.params)
        # test trained net without noise
        net_r = get_reward(model.params, env,human_data, None,render=True)
        mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
        mar_kid = kid_rewards.mean() if mar_kid is None else 0.9 * mar_kid + 0.1 * kid_rewards.mean()
        print(
            '| Net_R: %.1f' % mar,
            '| Kid_avg_R: %.1f' % kid_rewards.mean(),
            '| Gen_T: %.2f' % (time.time() - t0),
            '| test reward: %.5f' % net_r)

        if g % 100 == 0:
            print('saving model...')
            model.model.save_weights(GAME+".h5", overwrite=True)
