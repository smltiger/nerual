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



def sign(k_id): return -1. if k_id % 2 == 0 else 1.



class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v

def get_reward(block_params, env, seed_and_id=None, render=False):
    global model
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        block_params += sign(k_id) * SIGMA * np.random.randn(block_params.size)
    
    # run episode
    s = env.reset()
    s = skimage.color.rgb2gray(s)
    s = s[35:175,:]
    s = feature.canny(s)
    s = s.reshape(1,s.shape[0],s.shape[1],1)
    ep_r = 0.
    done = False
    action = [0,2,3]
    
    while not done:
        predict = model.model.predict(s)
        action_index = np.argmax(predict)
        s, r, done, _ = env.step(action[action_index])
        s = skimage.color.rgb2gray(s)
        s = s[35:175,:]
        s = feature.canny(s)
        s = s.reshape(1,s.shape[0],s.shape[1],1)
        if render:
            env.render()
        if r < 0:
            done = True
        elif r == 0:
            r = 0.01
        ep_r += r
    
    return ep_r


class ESModel():
    def __init__(self):
        model = Sequential()
        
        model.add(Convolution2D(20,(7,7),strides=(3,3),padding='same',input_shape=(140,160,1),use_bias=False,name='1'))
        model.add(Activation('tanh'))

        model.add(Convolution2D(20,(5,5),strides=(3,3),padding='same',use_bias=False,name='2'))
        model.add(Activation('tanh'))

        model.add(Convolution2D(30,(4,4),strides=(2,2),padding='same',use_bias=False,name='3'))
        model.add(Activation('tanh'))
        
        model.add(Convolution2D(30,(4,4),strides=(2,2),padding='same',use_bias=False,name='4'))
        model.add(Activation('tanh'))

        model.add(Convolution2D(50,(3,3),strides=(1,1),padding='same',use_bias=False,name='5'))
        model.add(Activation('tanh'))

        model.add(Flatten())
            
        model.add(Dense(32,use_bias=False,name='6'))
        model.add(Activation('tanh'))
        model.add(Dense(3,use_bias=False,name='7'))

        self.model = model
        self.params = []
        self.layer_info = []
        block_size = 1000

        for layer in model.layers:
            if layer.name.isnumeric():  
                l = model.get_layer(name=layer.name)
                w = l.get_weights()
                self.layer_info.append([l,w])
                if len(w) > 0:
                    w1 = w[0].reshape(w[0].size)
                    for k in range(int(w[0].size/block_size)+1):
                        param_block = w1[k*block_size:(k+1)*block_size]
                        if len(param_block) > 0:
                            self.params.append(param_block)

    def update_weights(self):
        for [layer,weight] in self.layer_info:
            layer.set_weights(weight)
        



def train(block_params, optimizer, utility, pool):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)    # mirrored sampling
    
    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, (block_params, env,
                                          [noise_seed[k_id], k_id],False)) for k_id in range(N_KID*2)]
    
    rewards = np.array([j.get() for j in jobs])
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward
    print(rewards)

    cumulative_update = np.zeros_like(block_params)       # initialize update values
    for ui, k_id in enumerate(kids_rank):
        np.random.seed(noise_seed[k_id])                # reconstruct noise using seed
        cumulative_update += utility[ui] * sign(k_id) * np.random.randn(block_params.size)

    gradients = optimizer.get_gradients(cumulative_update/(2*N_KID*SIGMA))
    return gradients, rewards


np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=160)
np.set_printoptions(formatter={'float':'{:0.3f}'.format})

N_KID = 10                  # half of the training population
N_GENERATION = 5000000         # training step
LR = .05                    # learning rate
SIGMA = .05                 # mutation strength or step size
N_CORE = mp.cpu_count()-1
model = ESModel()
model.model.load_weights('model.hd5')

if __name__ == "__main__":
    # utility instead reward for update parameters (rank transformation)
    base = N_KID * 2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base
    # training
    
    env = gym.make('Pong-v0')
    pool = mp.Pool(processes=N_CORE)
    mar = None      # moving average reward
    for g in range(N_GENERATION):
        print('*******Gen:',g)
        for block_params in model.params:
            t0 = time.time()
            optimizer = SGD(block_params, LR)
            gradients,kid_rewards = train(block_params, optimizer, utility, pool)
            block_params += gradients
            model.update_weights()
            # test trained net without noise
            net_r = get_reward(block_params, env, None,render=True)
            mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
            print(
                '| Net_R: %.1f' % mar,
                '| Kid_avg_R: %.1f' % kid_rewards.mean(),
                '| Gen_T: %.2f' % (time.time() - t0),
                '| test reward: %.5f' % net_r)
