import numpy as np
import torch
from qmix_vh_dis import QMIX
from transformers import AutoModelForCausalLM, LlamaTokenizer
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class Agents:
    def __init__(self, args):
        self.num_actions = args.num_actions
        self.num_agents = args.num_agents
        self.state_space = args.state_space
        self.obs_space = args.obs_space
        self.action_space = args.action_space
        self.policy = QMIX(args)
        self.args = args
        self.model = model_WithValueHead

    def choose_action(self, obs, agent_num, action, epsilon, evaluate=False):
        inputs = obs.copy()
        agent_id = torch.zeros(self.num_agents).long()
        agent_id[agent_num] = 1.
        inputs = torch.tensor(inputs).unsqueeze(0)
        hidden_state = self.policy.eval_hidden[:, agent_num, :]
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # 以概率ε选择一个随机动作（探索），以概率1-ε选择当前估计最优的动作（利用）
        attention_mask = torch.tensor([1] * len(inputs)).unsqueeze(0).cuda().to(torch.long)
        if np.random.uniform() < epsilon:
            action = torch.randint(0, self.args.action_space, (1,  self.args.num_actions)).unsqueeze(0)
        else:
            action = action
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]

        max_episode_len = 0

        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.max_episode_steps):
                if transition_idx + 1 >= max_episode_len:
                    max_episode_len = transition_idx + 1
                break

        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        self.policy.learn(batch, train_step, self.model)
        if train_step > 0 and ((train_step+1) % self.args.save_cycle == 0):

            self.policy.save_model(train_step)
