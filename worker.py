from parameters import *
from env.task_env import TaskEnv
import copy
import torch
import random
import numpy as np
from torch.nn import functional as F
from torch.distributions import Categorical
import time

class AdaptiveRewardWeights:
    """动态调整奖励权重"""
    def __init__(self):
        self.weights = {
            'time': 1.0,
            'efficiency': 1.0,
            'collaboration': 0.5,
            'success': 2.0
        }
        self.performance_history = []
        self.update_frequency = 100  # 每100个episode更新一次
    
    def update_weights(self, episode_performance):
        """根据训练表现动态调整权重"""
        self.performance_history.append(episode_performance)
        
        if len(self.performance_history) >= self.update_frequency:
            recent_performance = np.mean([p['success_rate'] for p in self.performance_history[-50:]])
            recent_efficiency = np.mean([p['efficiency'] for p in self.performance_history[-50:]])
            
            # 如果成功率低，增加成功率权重
            if recent_performance < 0.8:
                self.weights['success'] = min(self.weights['success'] * 1.05, 3.0)
            
            # 如果效率低，增加协作权重
            if recent_efficiency > 2.0:
                self.weights['collaboration'] = min(self.weights['collaboration'] * 1.05, 2.0)
            
            # 权重归一化
            total_weight = sum(self.weights.values())
            for key in self.weights:
                self.weights[key] = (self.weights[key] / total_weight) * 4.0  # 保持总权重为4.0
    
    def get_weights(self):
        """获取当前权重"""
        return self.weights.copy()

class Worker:
    def __init__(self, mete_agent_id, local_network, local_baseline, global_step, device='cuda', seed=None, env_params=None):
        self.device = device
        self.metaAgentID = mete_agent_id
        self.global_step = global_step
        if env_params is None:
            env_params = [EnvParams.SPECIES_AGENTS_RANGE, EnvParams.SPECIES_RANGE, EnvParams.TASKS_RANGE]
        self.env = TaskEnv(*env_params, traits_dim=EnvParams.TRAIT_DIM, max_task_size=2, duration_scale=5, seed=seed, single_ability=False, heterogeneous_speed=False)
        self.baseline_env = copy.deepcopy(self.env)
        self.local_baseline = local_baseline
        self.local_net = local_network
        self.experience = {idx:[] for idx in range(7)}
        self.episode_number = None
        self.perf_metrics = {}
        self.max_time = EnvParams.MAX_TIME

        # 初始化自适应权重
        self.reward_weights = AdaptiveRewardWeights()
        self.step_count = 0

    def convert_torch(self, args):
        data = []
        for arg in args:
            data.append(torch.tensor(arg, dtype=torch.float).to(self.device))
        return data

    @staticmethod
    def obs_padding(task_info, agents, mask):
        task_info = F.pad(task_info, (0, 0, 0, EnvParams.TASKS_RANGE[1] + 1 - task_info.shape[1]), 'constant', 0)
        agents = F.pad(agents, (0, 0, 0, EnvParams.SPECIES_AGENTS_RANGE[1] * EnvParams.SPECIES_RANGE[1] - agents.shape[1]), 'constant', 0)
        mask = F.pad(mask, (0, EnvParams.TASKS_RANGE[1] + 1 - mask.shape[1]), 'constant', 1)
        return task_info, agents, mask

    def run_episode(self, training=True, sample=False, max_waiting=False, cooperation=False, render=False, disable_mask=False):
        buffer_dict = {idx:[] for idx in range(7)}
        perf_metrics = {}
        current_action_index = 0
        decision_step = 0
        while not self.env.finished and self.env.current_time < EnvParams.MAX_TIME and current_action_index < 300:
            with torch.no_grad():
                release_agents, current_time = self.env.next_decision()
                self.env.current_time = current_time
                random.shuffle(release_agents[0])
                finished_task = []
                while release_agents[0] or release_agents[1]:
                    agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                    agent = self.env.agent_dic[agent_id]
                    task_info, total_agents, mask = self.convert_torch(self.env.agent_observe(agent_id, max_waiting=max_waiting, cooperation=cooperation, cooperation_threshold=1, disable_mask=disable_mask))
                    block_flag = mask[0, 1:].all().item()
                    if block_flag and not np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')):
                        agent['no_choice'] = block_flag
                        continue
                    elif block_flag and np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue
                    if training:
                        task_info, total_agents, mask = self.obs_padding(task_info, total_agents, mask)
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    probs, _ = self.local_net(task_info, total_agents, mask, index)
                    if training:
                        action = Categorical(probs).sample()
                        while action.item() > self.env.tasks_num:
                            action = Categorical(probs).sample()
                    else:
                        if sample:
                            action = Categorical(probs).sample()
                        else:
                            action = torch.argmax(probs, dim=1)
                    r, doable, f_t = self.env.agent_step(agent_id, action.item(), decision_step)
                    agent['current_action_index'] = current_action_index
                    finished_task.append(f_t)
                    if training and doable:
                        buffer_dict[0] += total_agents
                        buffer_dict[1] += task_info
                        buffer_dict[2] += action.unsqueeze(0)
                        buffer_dict[3] += mask
                        if TrainParams.USE_STEP_REWARDS:
                            buffer_dict[4] += torch.FloatTensor([[self.env.get_step_reward(agent_id, action.item())]]).to(self.device)  # reward
                        else:
                            buffer_dict[4] += torch.FloatTensor([[0]]).to(self.device)  # reward
                        buffer_dict[5] += index
                        buffer_dict[6] += torch.FloatTensor([[0]]).to(self.device)
                        current_action_index += 1
                self.env.finished = self.env.check_finished()
                decision_step += 1

        terminal_reward, finished_tasks = self.env.get_episode_reward(self.max_time)

        perf_metrics['success_rate'] = [np.sum(finished_tasks)/len(finished_tasks)]
        perf_metrics['makespan'] = [self.env.current_time]
        perf_metrics['time_cost'] = [np.nanmean(self.env.get_matrix(self.env.task_dic, 'time_start'))]
        perf_metrics['waiting_time'] = [np.mean(self.env.get_matrix(self.env.agent_dic, 'sum_waiting_time'))]
        perf_metrics['travel_dist'] = [np.sum(self.env.get_matrix(self.env.agent_dic, 'travel_dist'))]
        perf_metrics['efficiency'] = [self.env.get_efficiency()]
        return terminal_reward, buffer_dict, perf_metrics

    def run_episode_every_time(self, training=True, sample=False, max_waiting=False, cooperation=False, render=False, disable_mask=False):
        buffer_dict = {idx:[] for idx in range(7)}
        perf_metrics = {}

        # 添加死锁检测计数器
        deadlock_check_interval = TrainParams.DEADLOCK_CHECK_INTERVAL  # 每隔多少个时间步检查一次死锁
        deadlock_check_counter = 0

        while not self.env.finished and self.env.current_time < EnvParams.MAX_TIME:
            with torch.no_grad():
                # 渲染环境并处理退出请求
                if render:
                    quit_requested = self.env.render()
                    if quit_requested:
                        # 用户请求退出
                        print("用户请求退出仿真")
                        break
                # 如果暂停状态，则跳过更新逻辑
                if self.env.paused:
                    # 简单延时以避免CPU使用率过高
                    time.sleep(0.01)
                    continue
                release_agents = []
                for agent_id, agent in self.env.agent_dic.items():
                    # 跳过已禁用的智能体
                    if agent_id in self.env.disabled_agents:
                        continue
                    # 0时刻让所有智能体都加入决策
                    if self.env.current_time == 0:
                        release_agents.append(agent_id)
                        continue
                    # 如果智能体已经返回depot，不需要再做决策
                    if agent['current_task'] < 0:
                        if agent['returned']:
                            continue
                        else:
                            self.env.agent_step_every_time(agent_id, agent['current_task'])
                    elif agent['current_task'] >= 0 and self.env.task_dic[agent['current_task']]['finished']:
                        release_agents.append(agent_id)
                    elif agent['current_task'] >= 0 and not self.env.task_dic[agent['current_task']]['finished']:
                        self.env.agent_step_every_time(agent_id, agent['current_task'])

                # 定期检查死锁或在新增任务后检查，问题，有事强制重新分配不一定能完全解决问题，特别是面对argmax的情况，需要加一个掩码让他们不能选择目前选择的任务
                deadlock_check_counter += 1
                if deadlock_check_counter >= deadlock_check_interval:
                    deadlock_check_counter = 0
                    # 检查死锁并尝试解决
                    deadlock_detected = self.env.check_deadlock()
                    if deadlock_detected:
                        # 如果检测到死锁，尝试解决
                        if render:
                            print(f"时间 {self.env.current_time:.2f}: 检测到死锁")
                        
                        # 清空未完成任务的成员列表和相关状态
                        for task_id, task in self.env.task_dic.items():
                            # 跳过已经完成的任务
                            if task['finished']:
                                continue
                            if render and len(task['members']) > 0:
                                print(f"时间 {self.env.current_time:.2f}: 清空任务 {task_id} 的成员列表: {task['members']}")
                            task['members'] = []  # 清空成员列表
                            task['feasible_assignment'] = False  # 重置可行性标志
                            task['status'] = task['requirements'].copy()  # 重置状态为初始需求
                            task['abandoned_agent'] = []  # 清空放弃的智能体列表
                            task['sum_waiting_time'] = 0  # 重置等待时间
                            task['time_start'] = 0  # 重置开始时间
                            task['time_finish'] = 0  # 重置结束时间
                        
                        # 重置智能体的分配状态
                        for agent_id, agent in self.env.agent_dic.items():
                            if agent_id in self.env.disabled_agents:
                                continue
                            # 重置智能体的分配状态
                            agent['assigned'] = False
                            agent['sum_waiting_time'] = 0
                            release_agents.append(agent_id)
                            if render:
                                print(f"时间 {self.env.current_time:.2f}: 强制将智能体 {agent_id} 添加到决策队列")

                random.shuffle(release_agents)
                while release_agents:
                    agent_id = release_agents.pop(0)      
                    agent = self.env.agent_dic[agent_id]
                    task_info, total_agents, mask = self.convert_torch(self.env.agent_observe(agent_id=agent_id, max_waiting=max_waiting, cooperation=cooperation, cooperation_threshold=1, disable_mask=disable_mask))
                    block_flag = mask[0, 1:].all().item()
                    if block_flag and not np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')):
                        agent['no_choice'] = block_flag
                        continue
                    elif block_flag and np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue
                    if training:
                        task_info, total_agents, mask = self.obs_padding(task_info, total_agents, mask)
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    probs, _ = self.local_net(task_info, total_agents, mask, index)
                    if training:
                        action = Categorical(probs).sample()
                        while action.item() > self.env.tasks_num:
                            action = Categorical(probs).sample()
                    else:
                        if sample:
                            action = Categorical(probs).sample()
                        else:
                            action = torch.argmax(probs, dim=1)
                     # 执行选择的动作
                    if action.item() == 0:
                        # 返回仓库的动作是0（对应任务ID为-1）
                        depot_id = -agent['species'] - 1
                        agent['current_task'] = depot_id
                        self.env.agent_step_every_time(agent_id, depot_id)
                    else:
                        agent['current_task'] = action.item() - 1
                        self.env.agent_step_every_time(agent_id, agent['current_task'])
                    # 训练模式下存储经验
                    if training:
                        # buffer_dict[0]: 存储智能体的状态信息(total_agents)
                        buffer_dict[0] += total_agents
                        # buffer_dict[1]: 存储任务信息(task_info)
                        buffer_dict[1] += task_info
                        # buffer_dict[2]: 存储智能体选择的动作
                        buffer_dict[2] += action.unsqueeze(0)
                        # buffer_dict[3]: 存储动作掩码，用于标记哪些动作是可行的
                        buffer_dict[3] += mask
                        # buffer_dict[4]: 存储即时奖励，初始化为0
                        if TrainParams.USE_STEP_REWARDS:
                            buffer_dict[4] += torch.FloatTensor([[self.env.get_step_reward(agent_id, action.item())]]).to(self.device)  # reward
                        else:
                            buffer_dict[4] += torch.FloatTensor([[0]]).to(self.device)  # reward
                        # buffer_dict[5]: 存储智能体的ID
                        buffer_dict[5] += index
                        # buffer_dict[6]: 存储优势值，初始化为0
                        buffer_dict[6] += torch.FloatTensor([[0]]).to(self.device)
                # 更新环境时间
                self.env.current_time += self.env.dt
                # 检查是否完成
                self.env.check_finished_every_time()
        # 结束运行并收集结果
        terminal_reward, finished_tasks = self.env.get_episode_reward(self.max_time)
        perf_metrics['success_rate'] = [np.sum(finished_tasks)/len(finished_tasks)]
        perf_metrics['makespan'] = [self.env.current_time]
        perf_metrics['time_cost'] = [np.nanmean(self.env.get_matrix(self.env.task_dic, 'time_start'))]
        perf_metrics['waiting_time'] = [np.mean(self.env.get_matrix(self.env.agent_dic, 'sum_waiting_time'))]
        perf_metrics['travel_dist'] = [np.sum(self.env.get_matrix(self.env.agent_dic, 'travel_dist'))]
        perf_metrics['efficiency'] = [self.env.get_efficiency()]
        return terminal_reward, buffer_dict, perf_metrics

    def baseline_test(self):
        perf_metrics = {}
        current_action_index = 0
        start = time.time()
        while not self.baseline_env.finished and self.baseline_env.current_time < self.max_time and current_action_index < 300:
            with torch.no_grad():
                release_agents, current_time = self.baseline_env.next_decision()
                self.baseline_env.current_time = current_time
                random.shuffle(release_agents[0])
                if time.time() - start > 30:
                    break
                while release_agents[0] or release_agents[1]:
                    agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                    agent = self.baseline_env.agent_dic[agent_id]
                    task_info, total_agents, mask = self.convert_torch(self.baseline_env.agent_observe(agent_id, False))
                    return_flag = mask[0, 1:].all().item()
                    if return_flag and not np.all(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'feasible_assignment')): ## add condition on returning to depot
                        self.baseline_env.agent_dic[agent_id]['no_choice'] = return_flag
                        continue
                    elif return_flag and np.all(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue
                    task_info, total_agents, mask = self.obs_padding(task_info, total_agents, mask)
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    probs, _ = self.local_baseline(task_info, total_agents, mask, index)
                    action = torch.argmax(probs, 1)
                    self.baseline_env.agent_step(agent_id, action.item(), None)
                    current_action_index += 1
                self.baseline_env.finished = self.baseline_env.check_finished()

        reward, finished_tasks = self.baseline_env.get_episode_reward(self.max_time)
        return reward

    def work(self, episode_number):
        """
        Interacts with the environment. The agent gets either gradients or experience buffer
        """
        baseline_rewards = []
        buffers = []
        metrics = []
        max_waiting = TrainParams.FORCE_MAX_OPEN_TASK
        for _ in range(TrainParams.POMO_SIZE):
            self.env.init_state()

            # 根据参数选择使用哪种方法运行episode
            if TrainParams.USE_TIME_DRIVEN:
                terminal_reward, buffer, perf_metrics = self.run_episode_every_time(training=True, sample=True, max_waiting=max_waiting)
            else:
                terminal_reward, buffer, perf_metrics = self.run_episode(training=True, sample=True, max_waiting=max_waiting)

            if terminal_reward is np.nan:
                max_waiting = True
                continue
            baseline_rewards.append(terminal_reward)
            buffers.append(buffer)
            metrics.append(perf_metrics)

        # 计算基础平均奖励
        baseline_reward = np.nanmean(baseline_rewards)

        # 根据参数选择优势值计算方法
        if TrainParams.USE_ADAPTIVE_ADVANTAGE:
            # 使用改进的自适应优势值计算
            self.calculate_adaptive_advantage(buffers, baseline_rewards)
        else:
            # 使用原始的优势值计算方法
            for idx, buffer in enumerate(buffers):
                for key in buffer.keys():
                    if key == 6:
                        for i in range(len(buffer[key])):
                            if TrainParams.USE_STEP_REWARDS:
                                buffer[key][i] += buffer[4][i] + baseline_rewards[idx] - baseline_reward
                            else:
                                buffer[key][i] += baseline_rewards[idx] - baseline_reward
        
        # 更新自适应权重（仅在使用自适应优势值时）
        if TrainParams.USE_ADAPTIVE_ADVANTAGE and len(metrics) > 0:
            avg_metrics = {}
            for key in metrics[0].keys():
                avg_values = [m[key][0] if isinstance(m[key], list) and len(m[key]) > 0 else m[key] 
                             for m in metrics if key in m and m[key] is not None]
                if avg_values:
                    avg_metrics[key] = np.mean(avg_values)
                else:
                    avg_metrics[key] = 0.0
            
            # 确保包含所需的键
            if 'success_rate' not in avg_metrics:
                avg_metrics['success_rate'] = avg_metrics.get('success_rate', 0.8)
            if 'efficiency' not in avg_metrics:
                avg_metrics['efficiency'] = avg_metrics.get('efficiency', 1.0)
            
            self.reward_weights.update_weights(avg_metrics)

        # 整合经验到总经验缓冲区
        for idx, buffer in enumerate(buffers):
            for key in buffer.keys():
                if key not in self.experience.keys():
                    self.experience[key] = buffer[key]
                else:
                    self.experience[key] += buffer[key]

        # 整合性能指标
        for metric in metrics:
            for key in metric.keys():
                if key not in self.perf_metrics.keys():
                    self.perf_metrics[key] = metric[key]
                else:
                    self.perf_metrics[key] += metric[key]
                    
        self.episode_number = episode_number
        self.step_count += 1

    def calculate_adaptive_advantage(self, buffers, baseline_rewards):
        """自适应优势值计算"""
        baseline_reward = np.nanmean(baseline_rewards)
        
        for idx, buffer in enumerate(buffers):
            episode_reward = baseline_rewards[idx]
            
            # 基础优势值
            base_advantage = episode_reward - baseline_reward
            
            # 自适应调整因子
            performance_factor = self.calculate_performance_factor(buffer)
            exploration_bonus = self.calculate_exploration_bonus(buffer)
            
            # 改进的优势值
            for i in range(len(buffer[6])):  # buffer[6]是优势值
                # 时间衰减：早期决策影响更大
                time_decay = np.exp(-i * 0.01)
                
                # 动作重要性权重
                action_importance = self.calculate_action_importance(buffer, i)
                
                # 最终优势值
                adjusted_advantage = (base_advantage * time_decay * action_importance * performance_factor + exploration_bonus)
                if TrainParams.USE_STEP_REWARDS:
                    buffer[6][i] += buffer[4][i] + adjusted_advantage
                else:
                    buffer[6][i] += adjusted_advantage
    
    def calculate_performance_factor(self, buffer):
        """计算性能因子"""
        # 基于动作的多样性和质量计算性能因子
        if len(buffer[2]) == 0:  # buffer[2]是动作
            return 1.0
        
        actions = [action.item() if hasattr(action, 'item') else action for action in buffer[2]]
        
        # 动作多样性
        unique_actions = len(set(actions))
        diversity_factor = min(unique_actions / len(actions), 1.0)
        
        # 非零动作比例（选择任务而非返回depot）
        non_zero_actions = sum(1 for action in actions if action > 0)
        task_selection_factor = non_zero_actions / len(actions) if len(actions) > 0 else 0
        
        return 0.5 + 0.3 * diversity_factor + 0.2 * task_selection_factor
    
    def calculate_exploration_bonus(self, buffer):
        """计算探索奖励"""
        # 鼓励探索不同的任务和策略
        if len(buffer[2]) == 0:
            return 0
        
        actions = [action.item() if hasattr(action, 'item') else action for action in buffer[2]]
        
        # 探索奖励基于动作的熵
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        if len(action_counts) <= 1:
            return 0
        
        # 计算动作分布的熵
        total_actions = len(actions)
        entropy = 0
        for count in action_counts.values():
            prob = count / total_actions
            if prob > 0:
                entropy -= prob * np.log(prob)
        
        # 归一化熵值作为探索奖励
        max_entropy = np.log(len(action_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy * 0.1  # 小的探索奖励
    
    def calculate_action_importance(self, buffer, action_index):
        """计算动作重要性权重"""
        if action_index >= len(buffer[2]):
            return 1.0
        
        action = buffer[2][action_index]
        action_value = action.item() if hasattr(action, 'item') else action
        
        # 选择任务的动作比返回depot更重要
        if action_value > 0:
            return 1.2  # 任务选择动作更重要
        else:
            return 0.8  # 返回depot动作相对不重要