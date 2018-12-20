import queue

from multiprocessing.managers import BaseManager
from multiprocessing import freeze_support
from multiprocessing import Pool

from game import Game

import copy

import gym

import os

from graph import *


def sub_task():
    global task,result
    #每次从分布式队列task中选取SAMPLES个父样本
    SAMPLES = 6
    #将父样本扩展EXTEND倍
    EXTEND = 0
    HIGH_SCORE = 500
    
    env = gym.make('Pong-v0')
    env = env.unwrapped
    nets = []
    scores = {}
    pid = os.getpid()
    t = 0

    for n in range(SAMPLES):
        max_score = 0
        net = None
        for i in range(20):
            net = Graph([255,512,256,3])
            net.id = id(net)
            net.type = 'random'
            game = Game(env, net)
            score = game.run()
            if score > max_score:
                max_score = score
                max_net = net
        task.put(max_net)        
        
    while True:
        print('*********gerneration[',t,']*************')
        while len(nets) < SAMPLES * (EXTEND + 1):
            try:
                for i in range(SAMPLES):
                    net = task.get(timeout=1)
                    net.type = 'parent'
                    nets.append(net)
                    
                nets_tmp = []
                for i in range(len(nets)):
                    for j in range(i+1, len(nets)):
                        net = merge(nets[i], nets[j])
                        net.id = id(net)
                        net.type = 'child'
                        net.parent_score = [nets[i].score, nets[j].score]
                        nets_tmp.append(net)
                        for j in range(EXTEND):
                            net = copy.deepcopy(net)
                            net = net.mutate()
                            net.type = 'mutate'
                            nets_tmp.append(net)

                nets = nets + nets_tmp
            except queue.Empty:
                print('waiting for master nets...')

            for i in range(10):
                net = Graph([255,512,256,3])
                net.id = id(net)
                net.type = 'random'
                nets.append(net)
                
        print('-------generate ',len(nets),'nets')

        i = 0
        total_score = 0
        for net in nets:
            game = Game(env,net)
            scores[i] = game.run()
            net.score = scores[i]
            
            if net.score >= HIGH_SCORE:
                print('got a high score, retry 5 times')
                for k in range(5):
                    print('retry ',k+1)
                    net.score += game.run()
                scores[i] = net.score / 6
                net.score = scores[i]
                
            total_score += scores[i]
            print('net[',i,']pid:',pid,'running ',net.type,'net:',net.id,'parent_score',net.parent_score,' score:',scores[i],'params:',net.weight_count)
            i += 1

        average_score = total_score/i
        print('average score:',average_score)
        good_nets_list = sorted(scores.items(),key = lambda item:item[1], reverse=True)
        print('pid:',pid,'score:',good_nets_list)

        result_info = {}
        result_info['pid'] = pid
        result_info['gen'] = t
        result_info['best_score'] = []
        result_info['average_score'] = average_score

        for i in range(SAMPLES):
            idx = good_nets_list[i][0]
            task.put(nets[idx])
            print('pid:',pid,'put net:',nets[idx].id,'score:',good_nets_list[i][1],'params:',nets[idx].weight_count)
            result_info['best_score'].append(good_nets_list[i][1])

        result.put(result_info)
        nets.clear()
        scores.clear()
        t += 1
    
class QueueManager(BaseManager):
    pass

QueueManager.register('get_task_queue')
QueueManager.register('get_result_queue')
manager = QueueManager(address=('127.0.0.1',5000),authkey=b'abc')
manager.connect()
task = manager.get_task_queue()
result = manager.get_result_queue()


if __name__ == '__main__':
    '''CPU_CORES = 4
    freeze_support()
    p = Pool(CPU_CORES)
    for i in range(CPU_CORES):
        p.apply_async(sub_task)
    p.close()
    p.join()
    '''
    sub_task()
