import torch
from attention import AttentionNet
from parameters import EnvParams, TrainParams
from worker import Worker
import pickle
import time
import pandas as pd
from natsort import natsorted
import glob
from env.task_env import TaskEnv

USE_GPU = False
USE_GPU_GLOBAL = True
FOLDER_NAME = 'save'
testSet = 'RALTestSet'
model_path = f'model/{FOLDER_NAME}'
sampling = True
sampling_num = 1 if sampling else 1
max_task = False

def main(f):
    device = torch.device('cuda:0') if USE_GPU_GLOBAL else torch.device('cpu')
    global_network = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location=device, weights_only=False)
    global_network.load_state_dict(checkpoint['best_model'])
    worker = Worker(0, global_network, global_network, 0, device)
    index = int(f.split('/')[-1].replace('.pkl', '').replace('env_', '').replace(f'{testSet}\\', ''))
    env = pickle.load(open(f, 'rb'))
    results_best = None
    start = time.time()
    for i in range(sampling_num):
        env.init_state()
        worker.env = env
        _, _, results = worker.run_episode_every_time(False, sampling, max_task, render=True)
        # print(results)
        if results_best is None:
            results_best = results
        else:
            if results_best['makespan'] >= results['makespan']:
                results_best = results
    end = time.time() - start
    df_ = pd.DataFrame(results_best, index=[index])
    print(f"{worker.env.current_time:.1f}")
    return df_, end

if __name__ == "__main__":
    files = natsorted(glob.glob(f'{testSet}/env*.pkl'), key=lambda y: y.lower())
    main(files[52])