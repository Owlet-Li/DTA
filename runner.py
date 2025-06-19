import ray
from parameters import *
import torch
from attention import AttentionNet
from worker import Worker

class Runner(object):
    """Actor object to start running simulation on workers.
    Gradient computation is also executed on this object."""

    def __init__(self, metaAgentID):
        self.metaAgentID = metaAgentID
        self.device = torch.device('cuda') if TrainParams.USE_GPU else torch.device('cpu')
        self.localNetwork = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM)
        self.localNetwork.to(self.device)
        self.localBaseline = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM)
        self.localBaseline.to(self.device)

    def set_weights(self, weights):
        self.localNetwork.load_state_dict(weights)

    def set_baseline_weights(self, weights):
        self.localBaseline.load_state_dict(weights)

    def training(self, global_weights, baseline_weights, curr_episode, env_params):
        print("starting episode {} on metaAgent {}".format(curr_episode, self.metaAgentID))
        # set the local weights to the global weight values from the master network
        self.set_weights(global_weights)
        self.set_baseline_weights(baseline_weights)
        worker = Worker(mete_agent_id=self.metaAgentID, local_network=self.localNetwork, local_baseline=self.localBaseline, global_step=curr_episode, device=self.device, seed=None, env_params=env_params)
        worker.work(curr_episode)

        buffer = worker.experience
        perf_metrics = worker.perf_metrics

        info = {
            "id": self.metaAgentID,
            "episode_number": curr_episode,
        }

        return buffer, perf_metrics, info

    def testing(self, seed=None):
        worker = Worker(self.metaAgentID, self.localNetwork, self.localBaseline, 0, self.device, seed)
        reward = worker.baseline_test()
        return reward, seed, self.metaAgentID

@ray.remote(num_cpus=1, num_gpus=TrainParams.NUM_GPU / TrainParams.NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID):
        super().__init__(metaAgentID)