from parameters import *
from torch.utils.tensorboard import SummaryWriter
import os
import ray
import torch
from attention import AttentionNet
import torch.optim as optim
from runner import RLRunner
import numpy as np
from torch.distributions import Categorical

class Logger:
    def __init__(self):
        self.global_net = None
        self.baseline_net = None
        self.optimizer = None
        self.lr_decay = None
        self.writer = SummaryWriter(SaverParams.TRAIN_PATH)
        if SaverParams.SAVE:
            os.makedirs(SaverParams.MODEL_PATH, exist_ok=True)

    def set(self,  global_net, baseline_net, optimizer, lr_decay):
        self.global_net = global_net
        self.baseline_net = baseline_net
        self.optimizer = optimizer
        self.lr_decay = lr_decay

    def load_saved_model(self):
        print('Loading Model...')
        checkpoint = torch.load(SaverParams.MODEL_PATH + '/checkpoint.pth')
        if SaverParams.LOAD_FROM == 'best':
            model = 'best_model'
        else:
            model = 'model'
        self.global_net.load_state_dict(checkpoint[model])
        self.baseline_net.load_state_dict(checkpoint[model])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        curr_level = checkpoint['level']
        best_perf = checkpoint['best_perf']
        print("curr_episode set to ", curr_episode)
        print("best_perf so far is ", best_perf)
        print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        if TrainParams.RESET_OPT:
            self.optimizer = optim.Adam(self.global_net.parameters(), lr=TrainParams.LR)
            self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=TrainParams.DECAY_STEP, gamma=0.98)
        return curr_episode, curr_level, best_perf

    @staticmethod
    def generate_env_params(curr_level=None): # 生成环境参数，注意这个地方后面可以搞成课程学习，逐步加大环境复杂度
        per_species_num = np.random.randint(EnvParams.SPECIES_AGENTS_RANGE[0], EnvParams.SPECIES_AGENTS_RANGE[1] + 1)
        species_num = np.random.randint(EnvParams.SPECIES_RANGE[0], EnvParams.SPECIES_RANGE[1] + 1)
        tasks_num = np.random.randint(EnvParams.TASKS_RANGE[0], EnvParams.TASKS_RANGE[1] + 1)
        params = [(per_species_num, per_species_num), (species_num, species_num), (tasks_num, tasks_num)]
        return params

    @staticmethod
    def generate_test_set_seed():
        test_seed = np.random.randint(low=0, high=1e8, size=TrainParams.EVALUATION_SAMPLES).tolist()
        return test_seed

    def save_model(self, curr_episode, curr_level, best_perf):
        print('Saving model', end='\n')
        checkpoint = {"model": self.global_net.state_dict(),
                      "best_model": self.baseline_net.state_dict(),
                      "best_optimizer": self.optimizer.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "episode": curr_episode,
                      "lr_decay": self.lr_decay.state_dict(),
                      "level": curr_level,
                      "best_perf": best_perf
                      }
        path_checkpoint = "./" + SaverParams.MODEL_PATH + "/checkpoint.pth"
        torch.save(checkpoint, path_checkpoint)
        print('Saved model', end='\n')

    def write_to_board(self, tensorboard_data, curr_episode):
        tensorboard_data = np.array(tensorboard_data)
        tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
        reward, p_l, entropy, grad_norm, success_rate, time, time_cost, waiting, distance, effi = tensorboard_data
        metrics = {'Loss/Learning Rate': self.lr_decay.get_last_lr()[0],
                   'Loss/Policy Loss': p_l,
                   'Loss/Entropy': entropy,
                   'Loss/Grad Norm': grad_norm,
                   'Loss/Reward': reward,
                   'Perf/Makespan': time,
                   'Perf/Success rate': success_rate,
                   'Perf/Time cost': time_cost,
                   'Perf/Waiting time': waiting,
                   'Perf/Traveling distance':distance,
                   'Perf/Waiting Efficiency': effi
                   }
        for k, v in metrics.items():
            self.writer.add_scalar(tag=k, scalar_value=v, global_step=curr_episode)

def fuse_two_dicts(ini_dictionary1, ini_dictionary2):
    """
    合并两个字典，将相同键的值相加（列表连接）
    
    Args:
        ini_dictionary1: 第一个字典
        ini_dictionary2: 第二个字典，可以为None
    
    Returns:
        合并后的字典
    """
    if ini_dictionary2 is None:
        return ini_dictionary1
    
    final_dict = {}
    # 获取所有唯一的键
    all_keys = set(ini_dictionary1.keys()) | set(ini_dictionary2.keys())
    
    for key in all_keys:
        # 安全地获取每个字典中的值，如果键不存在则使用空列表
        value1 = ini_dictionary1.get(key, [])
        value2 = ini_dictionary2.get(key, [])
        # 将两个值相加（对于列表就是连接）
        final_dict[key] = value1 + value2
    
    return final_dict

def main():
    logger = Logger()
    ray.init()
    device = torch.device('cuda') if TrainParams.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if TrainParams.USE_GPU else torch.device('cpu')

    global_network = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    baseline_network = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    global_optimizer = optim.Adam(global_network.parameters(), lr=TrainParams.LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=TrainParams.DECAY_STEP, gamma=0.98)

    logger.set(global_network, baseline_network, global_optimizer, lr_decay)

    curr_episode = 0
    curr_level = 0
    best_perf = -200
    if SaverParams.LOAD_MODEL:
        curr_episode, curr_level, best_perf = logger.load_saved_model()

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(TrainParams.NUM_META_AGENT)]

    # get initial weights
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        baseline_weights = baseline_network.to(local_device).state_dict()
        global_network.to(device)
        baseline_network.to(device)
    else:
        weights = global_network.state_dict()
        baseline_weights = baseline_network.state_dict()
    weights_memory = ray.put(weights)
    baseline_weights_memory = ray.put(baseline_weights)

    # launch the first job on each runner
    jobs = []

    env_params = logger.generate_env_params(curr_level)
    for i, meta_agent in enumerate(meta_agents):
        jobs.append(meta_agent.training.remote(weights_memory, baseline_weights_memory, curr_episode, env_params))
        curr_episode += 1
    test_set = logger.generate_test_set_seed()
    baseline_value = None
    experience_buffer = {idx:[] for idx in range(7)}
    perf_metrics = {'success_rate': [], 'makespan': [], 'time_cost': [], 'waiting_time': [], 'travel_dist': [], 'efficiency': []}
    training_data = []

    try:
        while True:
            # wait for any job to be completed
            done_id, jobs = ray.wait(jobs)
            done_job = ray.get(done_id)[0]
            buffer, metrics, info = done_job
            experience_buffer = fuse_two_dicts(experience_buffer, buffer)
            perf_metrics = fuse_two_dicts(perf_metrics, metrics)

            update_done = False
            if len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                train_metrics = []
                while len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                    rollouts = {}
                    for k, v in experience_buffer.items():
                        rollouts[k] = v[:TrainParams.BATCH_SIZE]
                    for k in experience_buffer.keys():
                        experience_buffer[k] = experience_buffer[k][TrainParams.BATCH_SIZE:]
                    if len(experience_buffer[0]) < TrainParams.BATCH_SIZE:
                        update_done = True
                    if update_done:
                        for v in experience_buffer.values():
                            del v[:]

                    agent_inputs = torch.stack(rollouts[0], dim=0).to(device)  # (batch,sample_size,2)
                    task_inputs = torch.stack(rollouts[1], dim=0).to(device)  # (batch,sample_size,k_size)
                    action_batch = torch.stack(rollouts[2], dim=0).unsqueeze(1).to(device)  # (batch,1,1)
                    global_mask_batch = torch.stack(rollouts[3], dim=0).to(device)  # (batch,1,1)
                    reward_batch = torch.stack(rollouts[4], dim=0).unsqueeze(1).to(device)  # (batch,1,1)
                    index = torch.stack(rollouts[5]).to(device)
                    advantage_batch = torch.stack(rollouts[6], dim=0).to(device)  # (batch,1,1)

                    # REINFORCE
                    probs, _ = global_network(task_inputs, agent_inputs, global_mask_batch, index)
                    dist = Categorical(probs)
                    logp = dist.log_prob(action_batch.flatten())
                    entropy = dist.entropy().mean()
                    policy_loss = - logp * advantage_batch.flatten().detach()
                    policy_loss = policy_loss.mean()

                    loss = policy_loss
                    global_optimizer.zero_grad()

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=100, norm_type=2)
                    global_optimizer.step()

                    train_metrics.append([reward_batch.mean().item(), policy_loss.item(), entropy.item(), grad_norm.item()])
                lr_decay.step()

                perf_data = []
                for k, v in perf_metrics.items():
                    perf_data.append(np.nanmean(perf_metrics[k]))
                    del v[:]
                for v in perf_metrics.values():
                    del v[:]
                train_metrics = np.nanmean(train_metrics, axis=0)
                data = [*train_metrics, *perf_data]
                training_data.append(data)

            if len(training_data) >= TrainParams.SUMMARY_WINDOW:
                logger.write_to_board(training_data, curr_episode)
                training_data = []

            # get the updated global weights
            if update_done:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    baseline_weights = baseline_network.to(local_device).state_dict()
                    global_network.to(device)
                    baseline_network.to(device)
                else:
                    weights = global_network.state_dict()
                    baseline_weights = baseline_network.state_dict()
                weights_memory = ray.put(weights)
                baseline_weights_memory = ray.put(baseline_weights)

            env_params = logger.generate_env_params(curr_level)
            jobs.append(meta_agents[info['id']].training.remote(weights_memory, baseline_weights_memory, curr_episode, env_params))
            curr_episode += 1

            if curr_episode // (TrainParams.INCREASE_DIFFICULTY * (curr_level + 1)) == 1 and curr_level < 10:
                curr_level += 1
                print('increase difficulty to level', curr_level)

            if curr_episode % 512 == 0:
                logger.save_model(curr_episode, curr_level, best_perf)

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)

if __name__ == "__main__":
    main()