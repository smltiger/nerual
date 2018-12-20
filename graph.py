import random
import numpy as np
import copy


class Graph():
    def __init__(self, layers):
        layers_len = len(layers) - 1

        self.score = 0
        self.type = 'ancestor'
        self.parent_score = []
        self.id = 0
        
        self.graph = {}
        graph = self.graph
        self.v = []
        v = self.v

        self.layers = layers

        i = 0
        for layer in layers:
            graph[i] = []
            v.append([])
            for j in range(layer):
                #in,out,value
                graph[i].append([])
                v[i].append([])
            i += 1
            
        i = 0
        self.weight_count = 0
        for layer in layers:
            for j in range(layer):
                _layers = layers[i+1:]
                _i = i+1
                p = 0.01
                for _layer in _layers:
                    for _j in range(_layer):
                        if not (_i == layers_len and _i - i > 1):
                            if p > random.random():
                                w = (random.randint(0,20) - 10) * 0.06
                                node = graph[_i][_j]
                                node.append([i,j,w])
                                self.weight_count += 1
                    _i += 1
                    p = p * 0.1
            i += 1

    def forward(self, x):
        graph = self.graph
        layers = self.layers
        v = self.v

        for i in range(len(x)):
            v[0][i] = x[i]

        for i in range(1, len(layers)):
            for j in range(layers[i]):
                tmp = 0
                for w in graph[i][j]:
                    tmp += v[w[0]][w[1]] * w[2]
                if i < len(layers) - 1:
                    tmp = np.tanh(tmp)
                v[i][j] = tmp

        return v[len(layers) - 1]
        
    def mutate(self):
        graph = self.graph
        layers = self.layers
        layers_len = len(self.layers) - 1
        p = 0.01
        layer_prob = layers/np.sum(layers)

        t = 0
        while t < int(p*self.weight_count):
            tmp = np.random.choice(len(layers), 1, p=layer_prob)
            i = tmp[0]
            j = random.randint(0,layers[i] - 1)
            tmp = np.random.choice(len(layers), 1, p=layer_prob)
            _i = tmp[0]
            _j = random.randint(0,layers[_i] - 1)
            if i == _i:
                continue
            elif i > _i:
                tmp_i = i
                tmp_j = j
                i = _i
                j = _j
                _i = tmp_i
                _j = tmp_j
            found = None
            for node in graph[_i][_j]:
                if node[0] == i and node[1] == j:
                    found = node

            if found == None:
                w = (random.randint(0,20) - 10) * 0.06
                graph[_i][_j].append([i,j,w])
                self.weight_count += 1
            else:
                #op为3种情况：0-删除连接，1-增加w，2-减小w
                op = random.randint(0,2)
                if op == 0:
                    graph[_i][_j].remove(found)
                    self.weight_count -= 1
                elif op == 1:
                    found[2] += 0.01
                elif op == 2:
                    found[2] -= 0.01

            t += 1
            
        return self


def merge(net1, net2):
    net = copy.deepcopy(net1)
    graph = net.graph
    layers = net.layers
    layers_len = len(net.layers) - 1
    graph2 = net2.graph

    for i in range(1, len(layers)):
        for j in range(layers[i]):
            node2 = graph2[i][j]
            node = graph[i][j]
            for w2 in node2:
                k = 0
                for w in node:
                    if w2[0] == w[0] and w2[1] == w[1]:
                        w[2] += w2[2]
                        if abs(w[2]) < 0.01:
                            del node[k]
                            net.weight_count -= 1
                        elif abs(w[2]) > 0.7:
                            w[2] /= 2
                        k +=  1
                if k == 0:
                    node.append(w2)
                    net.weight_count += 1
    return net
        


