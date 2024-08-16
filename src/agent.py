import numpy as np
import torch

# __package__ = '.src.qmix_33-master'
from qmix_vh_dis import QMIX
from transformers import AutoModelForCausalLM, LlamaTokenizer
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = ""
print("數據集asdiv，模型：",model_id)
model_WithValueHead = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id,
    rope_scaling={
        "type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0, # 如果之前已经添加
        "high_freq_factor": 4.0,  # 如果之前已经添加
        "original_max_position_embeddings": 8192
    },
    device_map='auto',
    torch_dtype=torch.float16,
    max_length=300,
    eos_token_id=2,
    pad_token_id=2
)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)


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

    def choose_action(self, obs, agent_num, avail_actions, epsilon, evaluate=False):
        inputs = obs.copy()
        agent_id = torch.zeros(self.num_agents).long()
        agent_id[agent_num] = 1.
        inputs = torch.tensor(inputs).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()

        attention_mask = torch.tensor([1] * len(inputs)).unsqueeze(0).cuda().to(torch.long)
        with torch.no_grad():
            lm_logits, loss, value = model_WithValueHead.forward(input_ids=inputs,
                                                  attention_mask=attention_mask,
                                                  )

        if np.random.uniform() < epsilon:
            action = torch.randint(0, self.args.action_space, (1,  self.args.num_actions)).unsqueeze(0)
        else:
            action = torch.argmax(lm_logits, dim=2).unsqueeze(0)

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
