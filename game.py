import numpy as np

import skimage as skimage
from skimage import transform,color,exposure,io,feature

from keras import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *


class Game():
    def __init__(self, env, net):
        self.env = env
        self.net = net
        self.rows = 140
        self.cols = 160


        self.m = Sequential()

        m = self.m

        layer1 = Conv2D(1,(3,3),strides=3,padding='valid',input_shape=(self.rows,self.cols,1),use_bias=False)
        m.add(layer1)
        w = layer1.get_weights()
        w[0][0][0] = [[1.0]]
        w[0][0][1] = [[1.0]]
        w[0][0][2] = [[1.0]]
        w[0][1][0] = [[1.0]]
        w[0][1][1] = [[1.0]]
        w[0][1][2] = [[1.0]]
        w[0][2][0] = [[1.0]]
        w[0][2][1] = [[1.0]]
        w[0][2][2] = [[1.0]]
        layer1.set_weights(w)

        layer2 = Conv2D(1,(3,3),strides=3,padding='valid',use_bias=False)
        m.add(layer2)
        w = layer2.get_weights()
        w[0][0][0] = [[1.0]]
        w[0][0][1] = [[1.0]]
        w[0][0][2] = [[1.0]]
        w[0][1][0] = [[1.0]]
        w[0][1][1] = [[1.0]]
        w[0][1][2] = [[1.0]]
        w[0][2][0] = [[1.0]]
        w[0][2][1] = [[1.0]]
        w[0][2][2] = [[1.0]]
        layer2.set_weights(w)

        m.add(Flatten())


    def run(self):
        env = self.env
        net = self.net
        m = self.m
        
        s = env.reset()
        s = skimage.color.rgb2gray(s)
        s = s[35:175,:]
        s = s.reshape(1,self.rows,self.cols,1)
        s = m.predict(s)
        s = s.reshape(s.shape[1])
        done = False
        score = 0
        
        while not done:
            result = net.forward(s)
            idx = np.argmax(result)
            action = [0,2,3]
            s,r,done,info = env.step(action[idx])
            env.render()     

            s = skimage.color.rgb2gray(s)
            s = s[35:175,:]
            s = feature.canny(s)

            if r < 0:
                r = -1
                done = True
            elif r > 0:
                r = 0.1
            elif r == 0:
                r = 0.1

            #计算当前reward值

            ball = s[:, 20:137]
            bat = s[:, 138:145]
            ball_x,ball_y = np.where(ball > 0)
            bat_x, bat_y = np.where(bat > 0)
            if len(ball_x) == 0:
                ball_x = np.zeros(1)
                ball_x[0] = 70
            if len(bat_x) == 0:
                bat_x = np.zeros(1)
                ball_x[0] = 130
            r += (abs(np.average(ball_x) - np.average(bat_x)))*(-0.001)

            s = s.reshape(1,self.rows,self.cols,1)
            s = m.predict(s)
            s = s.reshape(s.shape[1])

            score += r

        return score
