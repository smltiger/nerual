import numpy as np
import gym
import random

from keras import *
from keras.layers import *

import skimage as skimage
from skimage import transform,color,exposure,io
from skimage import feature

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=160)
np.set_printoptions(formatter={'float':'{:0.3f}'.format})

model = Sequential()

layer1 = Conv2D(1,(3,1),strides=(3,1),padding='valid',input_shape=(140,160,1),use_bias=False)
model.add(layer1)
w = layer1.get_weights()
w[0][0][0] = [[1.0]]
#w[0][0][1] = [[1.0]]
#w[0][0][2] = [[1.0]]
w[0][1][0] = [[1.0]]
#w[0][1][1] = [[1.0]]
#w[0][1][2] = [[1.0]]
w[0][2][0] = [[1.0]]
#w[0][2][1] = [[1.0]]
#w[0][2][2] = [[1.0]]
layer1.set_weights(w)

layer2 = Conv2D(1,(1,3),strides=(1,3),padding='valid',use_bias=False)
model.add(layer2)
w = layer2.get_weights()
w[0][0][0] = [[1.0]]
w[0][0][1] = [[1.0]]
w[0][0][2] = [[1.0]]
#w[0][1][0] = [[1.0]]
#w[0][1][1] = [[1.0]]
#w[0][1][2] = [[1.0]]
#w[0][2][0] = [[1.0]]
#w[0][2][1] = [[1.0]]
#w[0][2][2] = [[1.0]]
layer2.set_weights(w)

layer3 = Conv2D(1,(2,1),strides=(2,1),padding='valid',use_bias=False)
model.add(layer3)
w = layer3.get_weights()
w[0][0][0] = [[1.0]]
#w[0][0][1] = [[1.0]]
#w[0][0][2] = [[1.0]]
w[0][1][0] = [[1.0]]
#w[0][1][1] = [[1.0]]
#w[0][1][2] = [[1.0]]
#w[0][2][0] = [[1.0]]
#w[0][2][1] = [[1.0]]
#w[0][2][2] = [[1.0]]
layer3.set_weights(w)

layer4 = Conv2D(1,(1,2),strides=(1,2),padding='valid',use_bias=False)
model.add(layer4)
w = layer4.get_weights()
w[0][0][0] = [[1.0]]
w[0][0][1] = [[1.0]]
#w[0][0][2] = [[1.0]]
#w[0][1][0] = [[1.0]]
#w[0][1][1] = [[1.0]]
#w[0][1][2] = [[1.0]]
#w[0][2][0] = [[1.0]]
#w[0][2][1] = [[1.0]]
#w[0][2][2] = [[1.0]]
layer4.set_weights(w)


env = gym.make('Pong-v0')
env.reset()

for i in range(20):
    env.step(0)
    
for i in range(100):
    s1,r1,done1,info1 = env.step(random.randint(0,5))
    s2,r2,done2,info2 = env.step(random.randint(0,5))

    print('*********')
    s1 = skimage.color.rgb2gray(s1)
    #s1 = skimage.transform.resize(s1, (105,80))
    s2 = skimage.color.rgb2gray(s2)
    #s2 = skimage.transform.resize(s2, (105,80))

    s1 = s1[35:175,:]
    
    a = feature.canny(s1)
    io.imshow(a)
    io.show()
 
    a = a.reshape(1,a.shape[0],a.shape[1],1)
    output = model.predict(a)
    output = output.reshape(output.shape[1],output.shape[2])
    io.imshow(output)
    io.show()

        
