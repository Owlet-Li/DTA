from parameters import EnvParams, TrainParams
import torch
from attention import AttentionNet
from worker import Worker
import pickle
import time
import copy
import pygame
import pandas as pd
import os
from natsort import natsorted
from env.task_env import TaskEnv
import glob

USE_GPU_GLOBAL = True

EnvParams.TRAIT_DIM = 5
TrainParams.AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM

FOLDER_NAME = 'save'
model_path = f'model/{FOLDER_NAME}'
testSet = 'RALTestSet'
sampling = True 
sampling_num = 10 if sampling else 1
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
        worker.baseline_env = copy.deepcopy(env)
        _, _, results = worker.run_episode(False, sampling, max_task)
        if results_best is None:
            results_best = results
        else:
            if results_best['makespan'] >= results['makespan']:
                results_best = results
    if EnvParams.RENDER_MODE == "human":
        print("开始pygame渲染...")
        # 确保pygame已初始化
        try:
            pygame.init()
        except:
            print("pygame初始化失败，请确保pygame已正确安装")
        worker.baseline_env.init_state()
        for j, agent in enumerate(worker.env.agent_dic.values()):
            agent_id = agent['ID']
            if len(agent['route']) > 1:
                worker.baseline_env.agent_dic[agent_id]['pre_set_route'] = agent['route'][1:]
                worker.baseline_env.agent_dic[agent_id]['current_task'] = -worker.baseline_env.agent_dic[agent_id]['species'] - 1
            else:
                worker.baseline_env.agent_dic[agent_id]['pre_set_route'] = []
        running = True
        while running and worker.baseline_env.current_time < EnvParams.MAX_TIME and not worker.baseline_env.finished:
            try:
                # 处理渲染和事件，检查是否请求退出
                quit_requested = worker.baseline_env.render()
                if quit_requested:
                    print("用户请求退出仿真")
                    running = False
                    break
                
                # 只在非暂停状态下更新环境
                if not worker.baseline_env.paused:
                    worker.baseline_env.execute_by_route_every_time()
                    worker.baseline_env.current_time += worker.baseline_env.dt
                else:
                    # 暂停状态下，简单延时以减少CPU使用
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"渲染过程中出现错误: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"渲染结束，环境状态: {'已完成' if worker.baseline_env.finished else '未完成'}")
        print(f"当前时间: {worker.baseline_env.current_time:.2f}/{EnvParams.MAX_TIME}")
        
        # 打印最终状态摘要
        finished_tasks = sum(1 for task in worker.baseline_env.task_dic.values() if task['finished'])
        total_tasks = len(worker.baseline_env.task_dic)
        returned_agents = sum(1 for agent in worker.baseline_env.agent_dic.values() if agent['returned'])
        total_agents = len(worker.baseline_env.agent_dic)
        print(f"最终状态: 任务完成 {finished_tasks}/{total_tasks}, 智能体返回 {returned_agents}/{total_agents}")
        
        # 清理pygame
        pygame.quit()
        
    end = time.time() - start
    df_ = pd.DataFrame(results_best, index=[index])
    print(f"测试完成: {f}")
    return df_, end

if __name__ == "__main__":
    # 确保存在测试集文件夹
    if not os.path.exists(testSet):
        try:
            os.makedirs(testSet)
        except:
            pass

    files = natsorted(glob.glob(f'{testSet}/env*.pkl'), key=lambda y: y.lower())
    
    # 检查是否有可用的测试环境文件
    if not files:
        print(f"创建测试环境文件...")
        os.makedirs(testSet, exist_ok=True)
        for i in range(1):  # 只创建一个环境进行测试
            env = TaskEnv((3, 3), (5, 5), (20, 20), 5, seed=i)
            with open(f'{testSet}/env_{i}.pkl', 'wb') as f:
                pickle.dump(env, open(f'{testSet}/env_{i}.pkl', 'wb'))
        files = natsorted(glob.glob(f'{testSet}/env*.pkl'), key=lambda y: y.lower())
    
    # 直接运行单个环境进行测试
    main(files[0])