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
import os

USE_GPU = False
USE_GPU_GLOBAL = True
FOLDER_NAME = 'save'
testSet = 'RALTestSet'
model_path = f'model/{FOLDER_NAME}'
training = False
sample = True
sampling_num = 1 if sample else 1
max_waiting = False
cooperation = True
render = True

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
        _, _, results = worker.run_episode_every_time(training=training, sample=sample, max_waiting=max_waiting, cooperation=cooperation, render=render)
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
    # 确保存在测试集文件夹
    if not os.path.exists(testSet):
        try:
            os.makedirs(testSet)
        except:
            pass

    # 指定要使用的测试环境序号
    i = 6
    target_file = f'{testSet}/env_{i}.pkl'
    
    # 检查指定序号的测试环境文件是否存在
    if not os.path.exists(target_file):
        print(f"创建测试环境文件 env_{i}.pkl...")
        os.makedirs(testSet, exist_ok=True)
        env = TaskEnv(per_species_range=(5, 10), species_range=(1, 5), tasks_range=(20, 20), traits_dim=5, decision_dim=10, max_task_size=5, duration_scale=1, seed=i, single_ability=True, heterogeneous_speed=True)
        with open(target_file, 'wb') as f:
            pickle.dump(env, f)
        print(f"测试环境文件 env_{i}.pkl 创建完成")
    else:
        print(f"找到已存在的测试环境文件 env_{i}.pkl")
    
    # 直接运行指定序号的环境进行测试
    main(target_file)