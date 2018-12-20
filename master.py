import queue
from multiprocessing.managers import BaseManager
from multiprocessing import freeze_support
from graph import Graph

def get_task_queue():
    global task_queue
    return task_queue

def get_result_queue():
    global result_queue
    return result_queue

class QueueManager(BaseManager):
    pass

task_queue = queue.Queue()
result_queue = queue.Queue()
SAMPLES = 6
CPU_CORES = 1
    
if __name__ == '__main__':
    freeze_support()
    QueueManager.register('get_task_queue', callable=get_task_queue)
    QueueManager.register('get_result_queue', callable=get_result_queue)
    manager = QueueManager(address=('127.0.0.1',5000),authkey=b'abc')
    manager.start()
    task = manager.get_task_queue()
    result = manager.get_result_queue()

    '''
    for i in range(SAMPLES * CPU_CORES):
        random_net = Graph([255,512,256,3])
        random_net.id = id(random_net)
        task.put(random_net)
        print('put an agent on task queue,id[',random_net.id,'] parameters:',random_net.weight_count)
    '''

    while True:
        try:
            info = result.get(timeout=10)
            print('current high fitness:',info)
        except queue.Empty:
            pass
    
