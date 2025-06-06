import numpy as np
import pygame
from pygame import gfxdraw
import time

class RenderSettings:
    """pygame渲染参数设置"""
    screen_width = 800
    screen_height = 800
    bg_color = (255, 255, 255)
    landmark_size = 16  # 任务标记大小
    agent_size = 8      # 智能体大小
    render_fps = 30     # 降低帧率使动画更流畅
    animation_speed = 1  # 动画速度调整系数，减小以放慢动画速度

class TaskEnv:
    def __init__(self, per_species_range=(10, 10), species_range=(5, 5), tasks_range=(30, 30), traits_dim=5, max_task_size=2, duration_scale=5, seed=None, single_ability=False, heterogeneous_speed=False):
        self.rng = None
        if seed is not None:
            self.rng = np.random.default_rng(seed) 
        self.per_species_range = per_species_range # 每个物种的智能体数量范围
        self.species_range = species_range # 物种数量范围
        self.tasks_range = tasks_range # 任务数量范围
        self.traits_dim = traits_dim # 能力值维度
        self.max_task_size = max_task_size # 每个任务的最大能力值
        self.duration_scale = duration_scale # 任务持续时间缩放因子
        self.single_ability = single_ability # 是否使用单一能力模式
        self.heterogeneous_speed = heterogeneous_speed # 是否使用速度异构模式

        self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = self.generate_env()
        self.species_distance_matrix, self.species_neighbor_matrix = self.generate_distance_matrix() 
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.species_num = len(self.species_dict['number'])

        self.current_time = 0
        self.dt = 0.1
        self.max_waiting_time = 200
        self.depot_waiting_time = 0
        self.finished = False

        # 添加一个列表用于存储被禁用的智能体
        self.disabled_agents = []
        self.screen = None
        self.clock = None
        self.surf = None
        self.paused = False
        self.species_colors = [
            (255, 0, 0),    # 红色
            (0, 0, 255),    # 蓝色
            (0, 255, 0),    # 绿色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
            (255, 128, 0),  # 橙色
            (128, 0, 255),  # 紫罗兰
            (0, 128, 255),  # 天蓝
            (128, 255, 0)   # 黄绿
        ]

    def random_int(self, low, high, size=None):
        if self.rng is not None:
            integer = self.rng.integers(low, high, size)
        else:
            integer = np.random.randint(low, high, size)
        return integer

    def random_value(self, row, col):
        if self.rng is not None:
            value = self.rng.random((row, col))
        else:
            value = np.random.rand(row, col)
        return value

    def generate_env(self):
        tasks_num = self.random_int(self.tasks_range[0], self.tasks_range[1] + 1) # 任务数量
        species_num = self.random_int(self.species_range[0], self.species_range[1] + 1) # 物种数量
        agents_species_num = [self.random_int(self.per_species_range[0], self.per_species_range[1] + 1) for _ in range(species_num)] # 每个物种的智能体数量

        agents_ini = self.generate_agent(species_num) # 每个物种的能力值
        tasks_ini = self.generate_task(tasks_num, species_num, agents_species_num) # 每个任务的能力值
        while not np.all(np.matmul(agents_species_num, agents_ini) >= tasks_ini):
            agents_ini = self.generate_agent(species_num)
            tasks_ini = self.generate_task(tasks_num, species_num, agents_species_num)

        # 生成每个物种的速度
        if self.heterogeneous_speed:
            # 速度异构模式：为每个物种生成不同的速度，范围在0.1到0.4之间
            species_velocities = self.random_value(species_num, 1).flatten() * 0.3 + 0.1  # 范围[0.1, 0.4]
        else:
            # 原始模式：所有物种速度相同
            species_velocities = np.full(species_num, 0.2)

        depot_loc = self.random_value(species_num, 2)
        cost_ini = [self.random_value(1, 1) for _ in range(species_num)]
        tasks_loc = self.random_value(tasks_num, 2)
        tasks_time = self.random_value(tasks_num, 1) * self.duration_scale

        task_dic = dict()
        agent_dic = dict()
        depot_dic = dict()
        species_dict = dict()
        species_dict['abilities'] = agents_ini
        species_dict['number'] = agents_species_num
        species_dict['velocities'] = species_velocities

        for i in range(tasks_num):
            task_dic[i] = {'ID': i,                                    # 任务的唯一标识符
                           'requirements': tasks_ini[i, :],            # 任务的能力需求向量，每个维度表示对应能力的需求量
                           'members': [],                              # 分配给此任务的智能体ID列表
                           'cost': [],                                 # 每个分配智能体的成本列表（暂未使用）
                           'location': tasks_loc[i, :],               # 任务在2D空间中的位置坐标 [x, y]
                           'feasible_assignment': False,              # 布尔值，表示任务是否已找到可行的智能体分配方案
                           'finished': False,                         # 布尔值，表示任务是否已完成
                           'time_start': 0,                           # 任务实际开始执行的时间
                           'time_finish': 0,                          # 任务完成的时间
                           'status': tasks_ini[i, :],                 # 任务当前状态，表示各维度剩余的能力需求量
                           'time': float(tasks_time[i, :]),           # 任务执行所需的持续时间
                           'sum_waiting_time': 0,                     # 所有分配智能体在此任务上的总等待时间
                           'efficiency': 0,                           # 任务执行效率，用于评估任务完成质量
                           'abandoned_agent': [],                     # 被放弃分配的智能体ID列表（超时或其他原因）
                           'optimized_ability': None,                 # 优化后的能力分配（可选，用于高级算法）
                           'optimized_species': []}

        i = 0
        for s, n in enumerate(agents_species_num):
            species_dict[s] = []
            for j in range(n):
                agent_dic[i] = {'ID': i,                                    # 智能体的唯一标识符
                                'species': s,                               # 智能体所属的物种编号
                                'abilities': agents_ini[s, :],             # 智能体的能力向量，每个维度表示对应能力的数值
                                'location': depot_loc[s, :],               # 智能体当前在2D空间中的位置坐标 [x, y]
                                'route': [- s - 1],                       # 智能体的路径规划，负数表示仓库ID，正数表示任务ID
                                'current_task': - s - 1,                  # 智能体当前执行的任务ID，负数表示在仓库中
                                'contributed': False,                      # 布尔值，表示智能体是否对当前任务做出了贡献
                                'arrival_time': [0.],                      # 智能体到达路径中各个位置的时间列表
                                'cost': cost_ini[s],                       # 智能体的运营成本
                                'travel_time': 0,                          # 智能体当前行程的旅行时间
                                'velocity': species_velocities[s],         # 智能体的移动速度（基于物种）
                                'next_decision': 0,                        # 智能体下一次需要做决策的时间点
                                'depot': depot_loc[s, :],                  # 智能体所属仓库的位置坐标
                                'travel_dist': 0,                          # 智能体累计的旅行距离
                                'sum_waiting_time': 0,                     # 智能体累计的等待时间
                                'current_action_index': 0,                 # 智能体当前动作在动作序列中的索引
                                'decision_step': 0,                        # 智能体当前的决策步数
                                'task_waiting_ratio': 1,                   # 任务等待时间比例因子
                                'trajectory': [],                          # 智能体的历史轨迹记录
                                'angle': 0,                                # 智能体当前的朝向角度
                                'returned': False,                         # 布尔值，表示智能体是否已返回仓库
                                'assigned': False,                         # 布尔值，表示智能体是否被分配给某个任务
                                'pre_set_route': None,                     # 预设的路径（如果有的话）
                                'no_choice': False}                        # 布尔值，表示智能体是否无可选择的动作
                species_dict[s].append(i)
                i += 1

        for s in range(species_num):
            depot_dic[s] = {'location': depot_loc[s, :],               # 仓库在2D空间中的位置坐标 [x, y]
                            'members': species_dict[s],                # 属于此仓库的智能体ID列表
                            'ID': - s - 1}                            # 仓库的唯一标识符（负数以区分任务ID）

        return task_dic, agent_dic, depot_dic, species_dict

    def generate_agent(self, species_num):
        if self.single_ability:
            # 单一能力模式：每个物种按顺序分配一种能力
            agents_ini = np.zeros((species_num, self.traits_dim), dtype=int)
            # 确保有足够的维度给每个物种分配不同的能力
            if species_num > self.traits_dim:
                raise ValueError(f"物种数量 ({species_num}) 不能超过能力维度 ({self.traits_dim})")
            
            # 按顺序为每个物种分配能力：第0个物种[1,0,0,0,0]，第1个物种[0,1,0,0,0]，以此类推
            for i in range(species_num):
                agents_ini[i, i] = 1
        else:
            # 原有模式：可以有多种能力的组合
            agents_ini = self.random_int(0, 2, (species_num, self.traits_dim)) # 注意此处能力值只有0和1，为每个物种生成相应的能力
            while not np.all(np.sum(agents_ini, axis=1) != 0) or np.unique(agents_ini, axis=0).shape[0] != species_num: # 确保每个物种至少有一个能力，且每个物种的能力值都不同
                agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))
        return agents_ini

    def generate_task(self, tasks_num, species_num, agents_species_num=None):
        if self.single_ability:
            # 单一能力模式：先在前species_num个维度生成需求，然后扩展到self.traits_dim维度
            # 每个维度的需求不能超过该维度对应的智能体数量
            if agents_species_num is None:
                raise ValueError("在单一能力模式下，generate_task方法需要agents_species_num参数")
            
            tasks_ini_partial = np.zeros((tasks_num, species_num), dtype=int)
            for i in range(tasks_num):
                for j in range(species_num):
                    # 对第j个维度，需求值不能超过第j个物种的智能体数量和max_task_size的最小值
                    max_demand = min(agents_species_num[j], self.max_task_size)
                    tasks_ini_partial[i, j] = self.random_int(0, max_demand + 1)
            
            # 确保每个任务至少有一个需求
            while not np.all(np.sum(tasks_ini_partial, axis=1) > 0):
                for i in range(tasks_num):
                    if np.sum(tasks_ini_partial[i, :]) == 0:
                        # 如果这个任务没有需求，随机选择一个维度设置需求
                        dim = self.random_int(0, species_num)
                        max_demand = min(agents_species_num[dim], self.max_task_size)
                        if max_demand > 0:
                            tasks_ini_partial[i, dim] = self.random_int(1, max_demand + 1)
            
            # 扩展到self.traits_dim维度，剩余维度填充0
            if self.traits_dim > species_num:
                padding = np.zeros((tasks_num, self.traits_dim - species_num), dtype=int)
                tasks_ini = np.hstack([tasks_ini_partial, padding])
            else:
                tasks_ini = tasks_ini_partial
        else:
            # 原有模式：任务可以在任何维度上有需求
            tasks_ini = self.random_int(0, self.max_task_size + 1, (tasks_num, self.traits_dim))
            while not np.all(np.sum(tasks_ini, axis=1) != 0):
                tasks_ini = self.random_int(0, self.max_task_size + 1, (tasks_num, self.traits_dim))
        return tasks_ini

    def generate_distance_matrix(self):
        species_distance_matrix = {}
        species_neighbor_matrix = {}
        for species in range(len(self.species_dict['number'])):
            tmp_dic = {-1: self.depot_dic[species], **self.task_dic}
            distances = {}
            for from_counter, from_node in tmp_dic.items():
                distances[from_counter] = {}
                for to_counter, to_node in tmp_dic.items():
                    if from_counter == to_counter:
                        distances[from_counter][to_counter] = 0
                    else:
                        distances[from_counter][to_counter] = self.calculate_eulidean_distance(from_node, to_node)

            sorted_distance_matrix = {k: sorted(dist, key=lambda x: dist[x]) for k, dist in distances.items()}
            species_distance_matrix[species] = distances
            species_neighbor_matrix[species] = sorted_distance_matrix
        return species_distance_matrix, species_neighbor_matrix

    @staticmethod
    def calculate_eulidean_distance(agent, task):
        return np.linalg.norm(agent['location'] - task['location'])

    @staticmethod
    def get_matrix(dictionary, key):
        """
        :param key: the key to index
        :param dictionary: the dictionary for key to index
        """
        key_matrix = []
        for value in dictionary.values():
            key_matrix.append(value[key])
        return key_matrix

    def next_decision(self):
        decision_time = np.array(self.get_matrix(self.agent_dic, 'next_decision'))
        if np.all(np.isnan(decision_time)):
            return ([], []), max(map(lambda x: max(x) if x else 0, self.get_matrix(self.agent_dic, 'arrival_time')))
        no_choice = self.get_matrix(self.agent_dic, 'no_choice')
        decision_time = np.where(no_choice, np.inf, decision_time)
        next_decision = np.nanmin(decision_time)
        if np.isinf(next_decision):
            arrival_time = np.array([agent['arrival_time'][-1] for agent in self.agent_dic.values()])
            decision_time = np.where(no_choice, np.inf, arrival_time)
            next_decision = np.nanmin(decision_time)
        finished_agents = np.where(decision_time == next_decision)[0].tolist()
        blocked_agents = []
        for agent_id in np.where(np.isinf(decision_time))[0].tolist():
            if next_decision >= self.agent_dic[agent_id]['arrival_time'][-1]:
                blocked_agents.append(agent_id)
        release_agents = (finished_agents, blocked_agents)
        return release_agents, next_decision

    def agent_observe(self, agent_id, max_waiting=False, cooperation=False, cooperation_threshold=1, disable_mask=False):
        agent = self.agent_dic[agent_id]
        
        # 新增分支：如果启用disable_mask，直接返回全False的mask（所有任务都可选）
        if disable_mask:
            # 创建一个全False的mask，长度为任务数量（所有任务都可选择）
            mask = np.zeros(self.tasks_num, dtype=bool)
            # 添加depot选项（在最前面插入False）
            mask = np.insert(mask, 0, False)
            # 获取其他信息
            agents_info = np.expand_dims(self.get_current_agent_status(agent), axis=0)
            tasks_info = np.expand_dims(self.get_current_task_status(agent), axis=0)
            mask = np.expand_dims(mask, axis=0)
            return tasks_info, agents_info, mask
        
        # 正常的mask构建逻辑
        mask = self.get_unfinished_task_mask()
        contributable_mask = self.get_contributable_task_mask(agent_id)
        mask = np.logical_or(mask, contributable_mask)
        
        if max_waiting:
            waiting_tasks_mask, waiting_agents = self.get_waiting_tasks()
            waiting_len = np.sum(waiting_tasks_mask == 0)
            if waiting_len > 5:
                mask = np.logical_or(mask, waiting_tasks_mask)
        elif cooperation:
            # Cooperation分支：只有当该智能体能贡献的任务中等待任务数量超过阈值时才启用合作
            waiting_tasks_mask, waiting_agents = self.get_waiting_tasks()
            
            # 获取该智能体能够贡献的等待任务
            contributable_waiting_tasks = np.logical_and(~contributable_mask, ~waiting_tasks_mask)
            contributable_waiting_len = np.sum(contributable_waiting_tasks)
            
            # 只有当该智能体能贡献的等待任务超过阈值时，才将这些任务设为可选择（mask为False）
            if contributable_waiting_len > cooperation_threshold:
                # 让智能体能够贡献的等待任务变为可选择（取消掩码）
                mask[contributable_waiting_tasks] = False
                print(f"智能体 {agent_id} 启用合作模式：能贡献的等待任务数 = {contributable_waiting_len}")
            else:
                print(f"智能体 {agent_id} 未启用合作模式：能贡献的等待任务数 = {contributable_waiting_len} <= {cooperation_threshold}")
        
        mask = np.insert(mask, 0, False)
        agents_info = np.expand_dims(self.get_current_agent_status(agent), axis=0)
        tasks_info = np.expand_dims(self.get_current_task_status(agent), axis=0)
        mask = np.expand_dims(mask, axis=0)
        return tasks_info, agents_info, mask

    def get_unfinished_task_mask(self):
        mask = np.logical_not(self.get_unfinished_tasks())
        return mask

    def get_unfinished_tasks(self):
        unfinished_tasks = []
        for task in self.task_dic.values():
            unfinished_tasks.append(task['feasible_assignment'] is False and np.any(task['status'] > 0))
        return unfinished_tasks

    def get_contributable_task_mask(self, agent_id):
        agent = self.agent_dic[agent_id]
        contributable_task_mask = np.ones(self.tasks_num, dtype=bool)
        for task in self.task_dic.values():
            if not task['feasible_assignment']:
                ability = np.maximum(np.minimum(task['status'], agent['abilities']), 0.)
                if ability.sum() > 0:
                    contributable_task_mask[task['ID']] = False
        return contributable_task_mask

    def get_waiting_tasks(self):
        waiting_tasks = np.ones(self.tasks_num, dtype=bool)
        waiting_agents = []
        for task in self.task_dic.values():
            if not task['feasible_assignment'] and len(task['members']) > 0:
                waiting_tasks[task['ID']] = False
                waiting_agents += task['members']
        return waiting_tasks, waiting_agents

    def get_current_agent_status(self, agent):
        status = []
        for a in self.agent_dic.values():
            if a['current_task'] >= 0:
                current_task = a['current_task']
                arrival_time = self.get_arrival_time(a['ID'], current_task)
                travel_time = np.clip(arrival_time - self.current_time, a_min=0, a_max=None)
                if self.current_time <= self.task_dic[current_task]['time_start']:
                    current_waiting_time = np.clip(self.current_time - arrival_time, a_min=0, a_max=None)
                    remaining_working_time = np.clip(self.task_dic[current_task]['time_start'] + self.task_dic[current_task]['time'] - self.current_time, a_min=0, a_max=None)
                else:
                    current_waiting_time = 0
                    remaining_working_time = 0
            else:
                travel_time = 0
                current_waiting_time = 0
                remaining_working_time = 0
            temp_status = np.hstack([a['abilities'], travel_time, remaining_working_time, current_waiting_time, agent['location'] - a['location'], a['assigned']])
            status.append(temp_status)
        current_agents = np.vstack(status)
        return current_agents

    def get_arrival_time(self, agent_id, task_id):
        arrival_time = self.agent_dic[agent_id]['arrival_time']
        arrival_for_task = np.where(np.array(self.agent_dic[agent_id]['route']) == task_id)[0][-1]
        return float(arrival_time[arrival_for_task])

    def get_current_task_status(self, agent):
        status = []
        for t in self.task_dic.values():
            travel_time = self.calculate_eulidean_distance(agent, t) / agent['velocity']
            temp_status = np.hstack([t['status'], t['requirements'], t['time'], travel_time, agent['location'] - t['location'], t['feasible_assignment']])
            status.append(temp_status)
        status = [np.hstack([np.zeros(self.traits_dim), - np.ones(self.traits_dim), 0, self.calculate_eulidean_distance(agent, self.depot_dic[agent['species']]) / agent['velocity'], agent['location'] - agent['depot'], 1])] + status
        current_tasks = np.vstack(status)
        return current_tasks

    def agent_step(self, agent_id, task_id, decision_step):
        """
        :param agent_id: the id of agent
        :param task_id: the id of task
        :param decision_step: the decision step of the agent
        :return: end_episode, finished_tasks
        """
        #  choose any task
        task_id = task_id - 1
        if task_id != -1:
            agent = self.agent_dic[agent_id]
            task = self.task_dic[task_id]
            if task['feasible_assignment']:
                return -1, False, []
        else:
            agent = self.agent_dic[agent_id]
            task = self.depot_dic[agent['species']]
        agent['route'].append(task['ID'])
        previous_task = agent['current_task']
        agent['current_task'] = task_id
        travel_time = self.calculate_eulidean_distance(agent, task) / agent['velocity']
        agent['travel_time'] = travel_time
        agent['travel_dist'] += self.calculate_eulidean_distance(agent, task)
        if previous_task >= 0 and self.task_dic[previous_task]['time_finish'] < self.current_time:
            current_time = self.task_dic[previous_task]['time_finish']
        else:
            current_time = self.current_time
        agent['arrival_time'] += [current_time + travel_time]
        # calculate the angle from current location to next location
        agent['location'] = task['location']
        agent['decision_step'] = decision_step
        agent['no_choice'] = False

        if agent_id not in task['members']:
            task['members'].append(agent_id)
        f_t = self.task_update()
        self.agent_update()
        return 0, True, f_t

    def task_update(self):
        f_task = []
        # check each task status and whether it is finished
        for task in self.task_dic.values():
            if not task['feasible_assignment']:
                abilities = self.get_abilities(task['members'])
                arrival = np.array([self.get_arrival_time(member, task['ID']) for member in task['members']])
                task['status'] = task['requirements'] - abilities  # update task status
                # Agents will wait for the other agents to arrive
                if (task['status'] <= 0).all():
                    if np.max(arrival) - np.min(arrival) <= self.max_waiting_time:
                        task['time_start'] = float(np.max(arrival, keepdims=True))
                        task['time_finish'] = float(np.max(arrival, keepdims=True) + task['time'])
                        task['feasible_assignment'] = True
                        f_task.append(task['ID'])
                    else:
                        task['feasible_assignment'] = False
                        infeasible_members = arrival <= np.max(arrival, keepdims=True) - self.max_waiting_time
                        for member in np.array(task['members'])[infeasible_members]:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
                else:
                    task['feasible_assignment'] = False
                    for member in np.array(task['members']):
                        if self.current_time - self.get_arrival_time(member, task['ID']) >= self.max_waiting_time:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
            else:
                if self.current_time >= task['time_finish']:
                    task['finished'] = True

        # check depot status
        for depot in self.depot_dic.values():
            for member in depot['members']:
                if self.current_time >= self.get_arrival_time(member, depot['ID']) and np.all(self.get_matrix(self.task_dic, 'feasible_assignment')):
                    self.agent_dic[member]['returned'] = True
        return f_task

    def get_abilities(self, members):
        if len(members) == 0:
            return np.zeros(self.traits_dim)
        else:
            return np.sum(np.array([self.agent_dic[member]['abilities'] for member in members]), axis=0)

    def agent_update(self):
        for agent in self.agent_dic.values():
            if agent['current_task'] < 0:
                if np.all(self.get_matrix(self.task_dic, 'feasible_assignment')):
                    agent['next_decision'] = np.nan
                elif not np.isnan(agent['next_decision']):
                    agent['next_decision'] = np.inf
                else:
                    pass
            else:
                current_task = self.task_dic[agent['current_task']]
                if current_task['feasible_assignment']:
                    if agent['ID'] in current_task['members']:
                        agent['next_decision'] = float(current_task['time_finish'])
                        if self.current_time >= float(current_task['time_start']):
                            agent['assigned'] = True
                    else:
                        agent['next_decision'] = self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time
                        agent['assigned'] = False
                else:
                    agent['next_decision'] = self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time
                    agent['assigned'] = False

    def check_finished(self):
        self.task_update()
        decision_agents, current_time = self.next_decision()
        if len(decision_agents[0]) + len(decision_agents[1]) == 0:
            self.current_time = current_time
            finished = np.all(self.get_matrix(self.agent_dic, 'returned')) and np.all(self.get_matrix(self.task_dic, 'finished'))
        else:
            finished = False
        return finished

    def get_episode_reward(self, max_time=200):
        self.calculate_waiting_time()
        eff = self.get_efficiency()
        finished_tasks = self.get_matrix(self.task_dic, 'finished')
        dist = np.sum(self.get_matrix(self.agent_dic, 'travel_dist'))
        reward = - self.current_time - eff * 10 if self.finished else - max_time - eff * 10
        return reward, finished_tasks

    def calculate_waiting_time(self):
        for agent in self.agent_dic.values():
            agent['sum_waiting_time'] = 0
        for task in self.task_dic.values():
            arrival = np.array([self.get_arrival_time(member, task['ID']) for member in task['members']])
            if len(arrival) != 0:
                if task['feasible_assignment']:
                    task['sum_waiting_time'] = np.sum(np.max(arrival) - arrival) \
                                               + len(task['abandoned_agent']) * self.max_waiting_time
                else:
                    task['sum_waiting_time'] = np.sum(self.current_time - arrival) \
                                               + len(task['abandoned_agent']) * self.max_waiting_time
            else:
                task['sum_waiting_time'] = len(task['abandoned_agent']) * self.max_waiting_time
            for member in task['members']:
                if task['feasible_assignment']:
                    self.agent_dic[member]['sum_waiting_time'] += np.max(arrival) - self.get_arrival_time(member, task['ID'])
                else:
                    self.agent_dic[member]['sum_waiting_time'] += self.current_time - self.get_arrival_time(member, task['ID']) if self.current_time - self.get_arrival_time(member, task['ID']) > 0 else 0
            for member in task['abandoned_agent']:
                self.agent_dic[member]['sum_waiting_time'] += self.max_waiting_time
    
    def get_efficiency(self):
        for task in self.task_dic.values():
            if task['feasible_assignment']:
                task['efficiency'] = abs(np.sum(task['requirements'] - task['status'])) / task['requirements'].sum()
            else:
                task['efficiency'] = 10
        efficiency = np.mean(self.get_matrix(self.task_dic, 'efficiency'))
        return efficiency

    def init_state(self):
        for task in self.task_dic.values():
            task.update(members=[], cost=[], finished=False, status=task['requirements'],feasible_assignment=False,
                        time_start=0, time_finish=0, sum_waiting_time=0, efficiency=0, abandoned_agent=[])
        for agent in self.agent_dic.values():
            agent.update(route=[-agent['species'] - 1], location=agent['depot'], contributed=False,
                         next_decision=0, travel_time=0, travel_dist=0, arrival_time=[0.], assigned=False,
                         sum_waiting_time=0, current_action_index=0, decision_step=0, trajectory=[], angle=0,
                         returned=False, pre_set_route=None, current_task=-1, task_waiting_ratio=1, no_choice=False, next_action=0)
        for depot in self.depot_dic.values():
            depot.update(members=self.species_dict[-depot['ID'] - 1])
        self.current_time = 0
        self.max_waiting_time = 200
        self.finished = False
        self.paused = False
        # 重置禁用的智能体列表
        self.disabled_agents = []

    def _handle_agent_click(self, mouse_x, mouse_y):
        """处理鼠标点击智能体的事件，如果点击到智能体则返回True，否则返回False"""
        for agent_id, agent in self.agent_dic.items():
            # 跳过已禁用的智能体
            if agent_id in self.disabled_agents:
                continue
                
            # 计算智能体在屏幕上的位置
            agent_x = int(agent['location'][0] * RenderSettings.screen_width)
            agent_y = int((1 - agent['location'][1]) * RenderSettings.screen_height)
            
            # 计算点击位置与智能体位置之间的距离
            distance = np.sqrt((mouse_x - agent_x)**2 + (mouse_y - agent_y)**2)
            
            # 如果点击在智能体范围内
            if distance <= RenderSettings.agent_size + 5:  # 给一点容差
                # 将智能体标记为禁用
                self.disabled_agents.append(agent_id)
                print(f"智能体 {agent_id} 已被禁用")
                
                # 获取智能体当前所在任务
                current_task_id = agent['current_task']
                
                # 从所有任务中移除该智能体
                affected_tasks = []
                for task_id, task in self.task_dic.items():
                    if agent_id in task['members']:
                        task['members'].remove(agent_id)
                        affected_tasks.append(task_id)
                        print(f"从任务 {task_id} 中移除智能体 {agent_id}")
                
                # 重新评估受影响任务的状态
                for task_id in affected_tasks:
                    task = self.task_dic[task_id]
                    # 重新检查任务的可行性
                    if task['feasible_assignment']:
                        # 重新计算任务状态
                        abilities = self.get_abilities(task['members'])
                        task['status'] = task['requirements'] - abilities
                        
                        # 如果任务不再可行，更新状态
                        if not (task['status'] <= 0).all():
                            task['feasible_assignment'] = False
                            print(f"任务 {task_id} 变为不可行")
                        else:
                            # 如果任务仍然可行，重新计算开始和结束时间
                            arrival = np.array([self.get_arrival_time(member, task_id) for member in task['members']])
                            if len(arrival) > 0 and np.max(arrival) - np.min(arrival) <= self.max_waiting_time:
                                task['time_start'] = float(np.max(arrival, keepdims=True))
                                task['time_finish'] = float(np.max(arrival, keepdims=True) + task['time'])
                            else:
                                task['feasible_assignment'] = False
                                print(f"任务 {task_id} 变为不可行（等待时间过长）")
                
                # 如果智能体属于某个物种的depot成员，也从那里移除
                agent_species = agent['species']
                if agent_id in self.depot_dic[agent_species]['members']:
                    self.depot_dic[agent_species]['members'].remove(agent_id)
                
                # 将智能体设置为已返回状态，使其在检查环境完成状态时被忽略
                agent['returned'] = True
                
                # 每次只禁用一个智能体，找到后就退出
                return True
        return False

    def _create_new_task(self, mouse_x, mouse_y):
        """直接创建新任务，不检查是否为空白区域"""
        # 将鼠标位置转换为环境坐标
        location_x = mouse_x / RenderSettings.screen_width
        # 翻转y坐标转换
        location_y = 1 - (mouse_y / RenderSettings.screen_height)
        
        # 创建一个新任务
        new_task_id = self.tasks_num
        
        # 随机生成任务需求
        if self.single_ability:
            # 单一能力模式：基于未被禁用的智能体数量生成需求
            requirements = np.zeros(self.traits_dim, dtype=int)
            
            # 计算每个物种未被禁用的智能体数量
            available_agents_per_species = []
            for species_id in range(self.species_num):
                species_agents = self.species_dict[species_id]
                available_count = sum(1 for agent_id in species_agents if agent_id not in self.disabled_agents)
                available_agents_per_species.append(available_count)
            
            # 为前species_num个维度生成需求
            for i in range(self.species_num):
                if available_agents_per_species[i] > 0:
                    # 需求不能超过该维度可用的智能体数量
                    max_req = min(available_agents_per_species[i], self.max_task_size)
                    if max_req > 0:
                        requirements[i] = self.random_int(0, max_req + 1)
            
            # 确保至少有一个维度有需求
            if np.sum(requirements) == 0:
                # 找到有可用智能体的维度
                available_dims = [i for i in range(self.species_num) if available_agents_per_species[i] > 0]
                if available_dims:
                    # 随机选择一个维度设置需求
                    dim = self.random_int(0, len(available_dims))
                    selected_dim = available_dims[dim]
                    max_req = min(available_agents_per_species[selected_dim], self.max_task_size)
                    requirements[selected_dim] = self.random_int(1, max_req + 1)
            
            print(f"单一能力模式 - 可用智能体数量: {available_agents_per_species}, 生成需求: {requirements[:self.species_num]}")
        else:
            # 原有模式：根据物种能力总和设置任务需求，确保可完成
            species_abilities = self.species_dict['abilities']
            total_ability = np.sum(species_abilities, axis=0)
            
            # 创建一个随机需求，保证至少有一个维度有需求且不超过物种总能力的50%
            requirements = np.zeros(self.traits_dim, dtype=int)
            while np.sum(requirements) == 0:
                for i in range(self.traits_dim):
                    if total_ability[i] > 0:
                        max_req = max(1, int(total_ability[i] * 0.5))  # 最大需求为总能力的50%
                        requirements[i] = self.random_int(0, max_req + 1)
        
        # 创建任务
        self.task_dic[new_task_id] = {
            'ID': new_task_id,
            'requirements': requirements.copy(),  # 任务需求
            'members': [],  # 任务成员
            'cost': [],  # 成本
            'location': np.array([location_x, location_y]),  # 任务位置
            'feasible_assignment': False,  # 任务分配是否可行
            'finished': False,  # 任务是否完成
            'time_start': 0,  # 开始时间
            'time_finish': 0,  # 结束时间
            'status': requirements.copy(),  # 当前状态（未满足的需求）
            'time': float(self.random_value(1, 1)[0, 0] * self.duration_scale),  # 任务持续时间
            'sum_waiting_time': 0,  # 等待时间总和
            'efficiency': 0,  # 效率
            'abandoned_agent': [],  # 放弃的智能体
            'optimized_ability': None,  # 优化能力
            'optimized_species': []  # 优化物种
        }
        
        # 更新任务数量
        self.tasks_num += 1
        
        print(f"在位置({location_x:.2f}, {location_y:.2f})创建新任务 {new_task_id}，需求: {requirements}")
            
        return new_task_id

    def render(self):
        # 添加动画速度控制
        if not hasattr(self, 'last_render_time'):
            self.last_render_time = time.time()
        else:
            current_time = time.time()
            elapsed = current_time - self.last_render_time
            target_frame_time = 1.0 / (RenderSettings.render_fps * RenderSettings.animation_speed)
            
            # 如果时间间隔太短，则等待
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
            
            # 更新上次渲染时间
            self.last_render_time = time.time()

        # 使用hasattr和getattr来避免linter错误
        if hasattr(pygame, 'get_init') and not getattr(pygame, 'get_init')():
            if hasattr(pygame, 'init'):
                getattr(pygame, 'init')()

        # 确保总是处理所有事件，特别是在暂停状态下
        quit_requested = False
        for event in pygame.event.get():
            if hasattr(pygame, 'KEYDOWN') and event.type == getattr(pygame, 'KEYDOWN'):
                if hasattr(pygame, 'K_SPACE') and event.key == getattr(pygame, 'K_SPACE'):
                    # 空格键切换暂停状态
                    self.paused = not self.paused
                    print("暂停状态：", "已暂停" if self.paused else "运行中")
                elif hasattr(pygame, 'K_UP') and event.key == getattr(pygame, 'K_UP'):
                    # 加快动画速度
                    RenderSettings.animation_speed = min(5.0, RenderSettings.animation_speed * 1.2)
                    print(f"动画速度: {RenderSettings.animation_speed:.2f}")
                elif hasattr(pygame, 'K_DOWN') and event.key == getattr(pygame, 'K_DOWN'):
                    # 减慢动画速度
                    RenderSettings.animation_speed = max(0.1, RenderSettings.animation_speed / 1.2)
                    print(f"动画速度: {RenderSettings.animation_speed:.2f}")
                elif hasattr(pygame, 'K_ESCAPE') and event.key == getattr(pygame, 'K_ESCAPE'):
                    # ESC键退出
                    quit_requested = True
            elif hasattr(pygame, 'QUIT') and event.type == getattr(pygame, 'QUIT'):
                # 处理窗口关闭事件
                quit_requested = True
            # 处理鼠标点击事件，仅在暂停状态下有效
            elif hasattr(pygame, 'MOUSEBUTTONDOWN') and event.type == getattr(pygame, 'MOUSEBUTTONDOWN') and self.paused:
                # 获取鼠标点击位置
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                # 简化为左键点击即可，不再区分左右键
                if event.button == 1:  # 左键点击
                    # 先检测是否点击到智能体，如果点击到则禁用，否则创建任务
                    if not self._handle_agent_click(mouse_x, mouse_y):
                        # 如果没有点击到智能体，则创建新任务
                        self._create_new_task(mouse_x, mouse_y)

        if hasattr(pygame, 'event') and hasattr(pygame.event, 'pump'):
            pygame.event.pump()

        if self.screen is None:
            pygame.display.init()
            self.screen = pygame.display.set_mode((RenderSettings.screen_width, RenderSettings.screen_height))
            pygame.display.set_caption("HMRTA_MPE_ENV")
            self.screen.fill(RenderSettings.bg_color)

        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        if self.surf is None:
            self.surf = pygame.Surface((RenderSettings.screen_width, RenderSettings.screen_height))

        self.surf.fill(RenderSettings.bg_color)
        pygame.event.pump()
        
        myfont = pygame.font.Font(None, 20)
        small_font = pygame.font.Font(None, 16)
        
        # 渲染所有任务
        for task_id, task in self.task_dic.items():
            # 转换任务坐标为屏幕坐标 - 修改y坐标计算，使(0,0)位于左下角
            task_x = int(task['location'][0] * RenderSettings.screen_width)
            # 翻转y坐标
            task_y = int((1 - task['location'][1]) * RenderSettings.screen_height)
            # 根据任务状态选择不同颜色渲染
            if task['finished']:
                # 已完成任务 - 绿色
                gfxdraw.aacircle(self.surf, task_x, task_y, RenderSettings.landmark_size, (0, 125, 25))
                gfxdraw.filled_circle(self.surf, task_x, task_y, RenderSettings.landmark_size, (0, 125, 25))
                task_text = str(np.sum(task['status']))
                text_surface = myfont.render(task_text, True, (0, 125, 25))
            elif task['feasible_assignment']:
                # 有可行分配方案的任务 - 红色
                gfxdraw.aacircle(self.surf, task_x, task_y, RenderSettings.landmark_size, (125, 0, 25))
                gfxdraw.filled_circle(self.surf, task_x, task_y, RenderSettings.landmark_size, (125, 0, 25))
                task_text = str(np.sum(task['status']))
                text_surface = myfont.render(task_text, True, (125, 0, 25))
            else:
                # 无可行分配方案的任务 - 蓝色
                gfxdraw.aacircle(self.surf, task_x, task_y, RenderSettings.landmark_size, (0, 25, 125))
                gfxdraw.filled_circle(self.surf, task_x, task_y, RenderSettings.landmark_size, (0, 25, 125))
                task_text = str(np.sum(task['status']))
                text_surface = myfont.render(task_text, True, (0, 25, 125))
            text_rect = text_surface.get_rect(center=(task_x, task_y))
            self.surf.blit(text_surface, text_rect)
            
            # 显示任务当前status
            status_str = np.array2string(task['status'], precision=1, separator=',')
            status_text = small_font.render(status_str, True, (50, 50, 50))
            status_rect = status_text.get_rect(center=(task_x, task_y + RenderSettings.landmark_size + 15))
            self.surf.blit(status_text, status_rect)
            
        pygame.event.pump()
        # 渲染所有智能体（跳过已禁用的智能体）
        for agent_id, agent in self.agent_dic.items():
            # 跳过已禁用的智能体
            if agent_id in self.disabled_agents:
                continue
                
            # 使用当前计算的位置
            agent_x = int(agent['location'][0] * RenderSettings.screen_width)
            # 翻转y坐标
            agent_y = int((1 - agent['location'][1]) * RenderSettings.screen_height)
            
            # 根据智能体物种选择颜色
            species_color = self.species_colors[agent['species'] % len(self.species_colors)]
            if agent['assigned']:
                # 使用深色调表示已贡献的agent
                dark_color = tuple(max(0, c - 100) for c in species_color)
                # 确保颜色元组格式正确
                if len(dark_color) == 3:
                    gfxdraw.filled_circle(self.surf, agent_x, agent_y, RenderSettings.agent_size, dark_color)
                    gfxdraw.aacircle(self.surf, agent_x, agent_y, RenderSettings.agent_size, dark_color)
                else:
                    # 如果颜色格式不正确，使用默认深灰色
                    gfxdraw.filled_circle(self.surf, agent_x, agent_y, RenderSettings.agent_size, (100, 100, 100))
                    gfxdraw.aacircle(self.surf, agent_x, agent_y, RenderSettings.agent_size, (100, 100, 100))
            else:
                gfxdraw.filled_circle(self.surf, agent_x, agent_y, RenderSettings.agent_size, species_color)
                gfxdraw.aacircle(self.surf, agent_x, agent_y, RenderSettings.agent_size, species_color)
            # 在智能体上显示ID
            agent_text = str(agent_id)
            text_surface = myfont.render(agent_text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(agent_x, agent_y))
            self.surf.blit(text_surface, text_rect)
        
        # 渲染所有仓库
        for depot_id, depot in self.depot_dic.items():
            depot_x = int(depot['location'][0] * RenderSettings.screen_width)
            # 翻转y坐标
            depot_y = int((1 - depot['location'][1]) * RenderSettings.screen_height)
            
            # 获取对应物种的颜色
            species_color = self.species_colors[depot_id % len(self.species_colors)]
            
            # 绘制正方形depot
            square_size = RenderSettings.agent_size + 2
            rect = pygame.Rect(depot_x - square_size, depot_y - square_size, square_size * 2, square_size * 2)
            gfxdraw.box(self.surf, rect, species_color)
            gfxdraw.rectangle(self.surf, rect, species_color)
            
            # 显示仓库ID
            depot_text = str(depot_id)
            text_surface = myfont.render(depot_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(depot_x, depot_y))
            self.surf.blit(text_surface, text_rect)
            
            # 显示物种能力和速度
            species_id = depot_id
            if 0 <= species_id < len(self.species_dict['abilities']):
                ability_str = np.array2string(self.species_dict['abilities'][species_id], precision=1, separator=',')
                ability_text = small_font.render(ability_str, True, (50, 50, 50))
                ability_rect = ability_text.get_rect(center=(depot_x, depot_y + square_size + 15))
                self.surf.blit(ability_text, ability_rect)
                
                # 如果启用了速度异构，显示速度信息
                if self.heterogeneous_speed and 'velocities' in self.species_dict:
                    velocity_str = f"v:{self.species_dict['velocities'][species_id]:.2f}"
                    velocity_text = small_font.render(velocity_str, True, (0, 100, 0))
                    velocity_rect = velocity_text.get_rect(center=(depot_x, depot_y + square_size + 30))
                    self.surf.blit(velocity_text, velocity_rect)
        
        # 将绘制表面显示到屏幕
        self.screen.blit(self.surf, (0, 0))

        # 显示环境状态信息
        info_font = pygame.font.SysFont('microsoft yahei', 20)
        # 显示当前仿真时间
        time_text = info_font.render(f"已暂停" if self.paused else f"当前时间: {self.current_time:.2f}", True, (255, 0, 0) if self.paused else (0, 0, 0))
        self.screen.blit(time_text, (10, 10))
        # 显示任务完成情况
        finished_tasks = sum(1 for task in self.task_dic.values() if task['finished'])
        tasks_text = info_font.render(f"任务完成: {finished_tasks}/{len(self.task_dic)}", True, (0, 0, 0))
        self.screen.blit(tasks_text, (10, 35))
        
        # 显示智能体返回情况
        returned_agents = sum(1 for agent_id, agent in self.agent_dic.items() if agent['returned'] and agent_id not in self.disabled_agents)
        active_agents = len(self.agent_dic) - len(self.disabled_agents)
        agents_text = info_font.render(f"智能体返回: {returned_agents}/{active_agents}", True, (0, 0, 0))
        self.screen.blit(agents_text, (10, 60))
        
        # 显示禁用的智能体数量
        if self.disabled_agents:
            disabled_text = info_font.render(f"禁用智能体: {len(self.disabled_agents)}", True, (255, 0, 0))
            self.screen.blit(disabled_text, (10, 135))
        
        # 显示完成率
        if len(self.task_dic) > 0:
            task_completion_rate = finished_tasks / len(self.task_dic) * 100
            completion_text = info_font.render(f"任务完成率: {task_completion_rate:.1f}%", True, (0, 0, 0))
            self.screen.blit(completion_text, (10, 85))
        
        # 显示动画速度
        speed_text = info_font.render(f"动画速度: {RenderSettings.animation_speed:.2f}x", True, (0, 0, 0))
        self.screen.blit(speed_text, (10, 110))
        
        # 显示操作提示
        help_text = info_font.render("空格: 暂停/继续", True, (0, 0, 0))
        self.screen.blit(help_text, (RenderSettings.screen_width - 200, 35))

        help_text = info_font.render("上下键: 调整速度", True, (0, 0, 0))
        self.screen.blit(help_text, (RenderSettings.screen_width - 200, 60))
        
        # 如果处于暂停状态，显示鼠标点击提示
        if self.paused:
            click_text = info_font.render("点击智能体: 使其失效", True, (255, 0, 0))
            self.screen.blit(click_text, (RenderSettings.screen_width - 200, 85))
            
            task_text = info_font.render("点击空白处: 新增任务", True, (255, 0, 0))
            self.screen.blit(task_text, (RenderSettings.screen_width - 200, 110))
            
            # 使用字体显示暂停提示
            pause_font = pygame.font.SysFont('microsoft yahei', 36)
            pause_text = pause_font.render("已暂停", True, (255, 0, 0))
            text_rect = pause_text.get_rect(center=(RenderSettings.screen_width//2, 30))
            self.screen.blit(pause_text, text_rect)
        
        # 控制帧率
        if self.clock is not None:
            self.clock.tick(RenderSettings.render_fps)
        
        # 最后处理事件队列确保帧结束时界面响应
        pygame.event.pump()
        pygame.display.update()
        
        # 返回是否请求退出
        return quit_requested

    def execute_by_route_every_time(self):
        # 如果处于暂停状态，跳过更新
        if self.paused:
            return
            
        # 打印一些关键信息用于调试
        moving_agents = 0

        for agent_id, agent in self.agent_dic.items():
            # 调试输出当前任务和预设路由
            has_route = agent['pre_set_route'] is not None and len(agent['pre_set_route']) > 0
            if agent['current_task'] < 0:  # 当前在depot或无任务
                if has_route:
                    # 将下一个任务设为当前任务
                    next_task = agent['pre_set_route'].pop(0)
                    print(f"智能体 {agent_id} 从depot出发前往任务 {next_task}")
                    agent['current_task'] = next_task
                    # 重置返回状态，因为正在离开depot
                    agent['returned'] = False
                    moving_agents += 1
            else:  # 当前有任务
                task = self.task_dic[agent['current_task']]
                if task['finished']:  # 如果当前任务已完成
                    if has_route:
                        # 前往下一个任务
                        next_task = agent['pre_set_route'].pop(0)
                        print(f"智能体 {agent_id} 完成任务 {agent['current_task']}，前往任务 {next_task}")
                        agent['current_task'] = next_task
                        moving_agents += 1
                    else:
                        # 如果没有更多任务，返回depot
                        depot_id = -agent['species'] - 1
                        agent['current_task'] = depot_id
                        print(f"智能体 {agent_id} 完成所有任务，返回depot {depot_id}")
                        moving_agents += 1

            # 执行移动
            self.agent_step_every_time(agent_id, agent['current_task'])

        # 检查是否完成或出现死锁
        self.check_finished_every_time()

    def agent_step_every_time(self, agent_id, task_id):
        # 如果智能体已被禁用，则跳过处理
        if agent_id in self.disabled_agents:
            return 0, False, []
            
        agent = self.agent_dic[agent_id]
        
        # 处理任务ID，确定目标位置
        if task_id >= 0:
            # 选择了正常任务
            task = self.task_dic[task_id]
            target_location = task['location']
            is_depot = False
        else:
            # 选择返回depot，使用负数表示depot ID
            depot_species = agent['species']
            task = self.depot_dic[depot_species]
            target_location = task['location']
            is_depot = True
        
        if task['ID'] not in agent['route']:
            agent['route'].append(task['ID'])

        distance = self.calculate_eulidean_distance(agent, task)

        if agent_id not in task['members']:
            task['members'].append(agent_id)
            travel_time = distance / agent['velocity']
            agent['arrival_time'].append(self.current_time + travel_time)
            agent['travel_dist'] += distance
            
        
        # 移动智能体 - 使用更细粒度的移动
        if distance < self.dt * agent['velocity']:
            # 如果可以在当前时间步内到达目标
            agent['location'] = target_location
            
            # 如果是depot，标记为已返回
            if is_depot:
                agent['returned'] = True
                # print(f"智能体 {agent_id} 已返回depot")
                
                # 打印智能体返回状态
                returned_agents = sum(1 for a in self.agent_dic.values() if a['returned'])
                total_agents = len(self.agent_dic)
                # print(f"返回状态: {returned_agents}/{total_agents} 智能体已返回")
        else:
            # 部分移动向目标方向前进
            move_vector = target_location - agent['location']
            move_distance = self.dt * agent['velocity']
            
            # 单位化向量并乘以行进距离
            direction = move_vector / np.linalg.norm(move_vector)
            agent['location'] = agent['location'] + direction * move_distance
            
            # 如果是前往depot但尚未到达，确保不标记为已返回
            if is_depot:
                agent['returned'] = False
        
        # 更新任务状态
        self.task_update_every_time(task['ID'])
        return 0, True, []
    
    def task_update_every_time(self, task_id):
        if task_id < 0:
            return
        else:
            task = self.task_dic[task_id]
            if not task['feasible_assignment']:
                abilities = self.get_abilities(task['members'])
                arrival = np.array([self.get_arrival_time(member, task['ID']) for member in task['members']])
                task['status'] = task['requirements'] - abilities  # update task status
                # Agents will wait for the other agents to arrive
                if (task['status'] <= 0).all():
                    if np.max(arrival) - np.min(arrival) <= self.max_waiting_time:
                        task['time_start'] = float(np.max(arrival, keepdims=True))
                        task['time_finish'] = float(np.max(arrival, keepdims=True) + task['time'])
                        task['feasible_assignment'] = True
                    else:
                        task['feasible_assignment'] = False
                        infeasible_members = arrival <= np.max(arrival, keepdims=True) - self.max_waiting_time
                        for member in np.array(task['members'])[infeasible_members]:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
                else:
                    task['feasible_assignment'] = False
                    for member in np.array(task['members']):
                        if self.current_time - self.get_arrival_time(member, task['ID']) >= self.max_waiting_time:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
            else:
                if self.current_time >= task['time_finish']:
                    task['finished'] = True
    
    def check_finished_every_time(self):
        # 检查是否所有任务完成和所有未禁用智能体到达depot
        all_tasks_done = all(t['finished'] or all(t['status'] <= 0) for t in self.task_dic.values())
        all_agents_returned = all(a['returned'] or a_id in self.disabled_agents 
                                for a_id, a in self.agent_dic.items())
        # 更新环境完成状态
        if all_tasks_done and all_agents_returned:
            self.finished = True

    def check_deadlock(self):
        """检查是否出现死锁情况"""
        # 1. 检查是否存在未完成的任务
        unfinished_tasks = [t for t_id, t in self.task_dic.items() if not t['finished']]
        if not unfinished_tasks:
            return False  # 没有未完成任务，不存在死锁
        
        # 2. 检查所有智能体状态
        all_agents_waiting_or_returned = True
        for agent_id, agent in self.agent_dic.items():
            # 跳过被禁用的智能体
            if agent_id in self.disabled_agents:
                continue
            # 如果智能体正在执行任务，则不是死锁，feasible_assignment表示任务可行，此时不再死锁
            if agent['current_task'] >= 0 and self.task_dic[agent['current_task']]['feasible_assignment'] and not self.task_dic[agent['current_task']]['finished']:
                all_agents_waiting_or_returned = False
                break

        # 3. 死锁条件：有未完成任务，但所有智能体都在等待或已返回仓库
        return all_agents_waiting_or_returned and len(unfinished_tasks) > 0

    
if __name__ == '__main__':
    import pickle
    import os

    # 定义测试集目录名
    testSet = 'RALTestSet'

    # 获取当前文件所在目录的上一级目录的绝对路径
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

     # 创建目标目录（如果不存在）
    target_dir = os.path.join(base_dir, testSet)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目录: {target_dir}")

    # 创建环境对象
    i = 5
    # 可以选择是否启用速度异构：heterogeneous_speed=True 启用，heterogeneous_speed=False 或不设置则使用原始统一速度
    env = TaskEnv(per_species_range=(10, 10), species_range=(1, 1), tasks_range=(20, 20), traits_dim=5, max_task_size=1, duration_scale=10, seed=i, single_ability=True, heterogeneous_speed=True)
    # 保存环境对象到文件
    output_file = os.path.join(target_dir, f'env_{i}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(env, f)

    print(f"环境对象已保存到: {output_file}")