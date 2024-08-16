from gym.core import Env
import argparse
import numpy as np
# __package__ = '.src.qmix_33-master'
from utils import RewardFunction
from tqdm import tqdm
import torch

def qmix_args(args):
    args.rnn_hidden_dim = 64
    args.two_hyper_layers = False
    args.qmix_hidden_dim = 32
    args.lr = 2.9

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 500
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch to train the agent
    args.n_epoch = 10

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # the number of the train steps in one epoch
    args.train_steps = 1

    # # how often to evaluate
    args.evaluate_cycle = 10

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 10

    # how often to update the target_net
    args.target_update_cycle = 10

    # prevent gradient explosion
    args.grad_norm_clip = 10

    #新加：
    args.reward_scale_rate = 20
    args.merged_dataset = "/home/lihaoran/lhr/maq/saved_data/reward/merged_dataset.json"
    args.standard_answer = []


    return args


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_space', type=int, default=256, help='observation space')


    parser.add_argument('--state_space', type=int, default=1000, help='state_space')


    parser.add_argument('--action_space', type=int, default=32000, help='action space:因为文本数据的空间是连续的，动作空间应该是连续的；'
                                                                    '使用的是一个rnn，其输出层有一个大小为词汇表大小的softmax层，这里每个单元对应一个词汇表中的单词，输出层的尺寸就是词汇表的大小，即 action_space')
    parser.add_argument('--num_actions', type=int, default=256, help='智能体至少需要能够执行两种动作（初始回答和基于其他智能体回答的更新;  因为环境要根据动作计算奖励，要把这个改为环境根据回答答案计算奖励')
    parser.add_argument('--num_agents', type=int, default=3, help='number of agents')
    parser.add_argument('--max_episode_steps', type=int, default=1, help='最大episode步长')


    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=2, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='the directory of save the replay')

    parser.add_argument('--alg', type=str, default='qmix_', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=False, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=False, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--n_evaluate_episode', type=int, default=3, help='number of the episode to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')

    parser.add_argument('--load_model', type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--load_model_before', type=bool, default=False, help='whether to load the pretrained model')

    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--threshold', type=float, default=19.9, help='threshold to judge whether win')


    parser.add_argument('--standard_answer', type=dict, default=[], help='标准答案')
    parser.add_argument('--observation', type=dict, default=[], help='观测值即"role": "user", "content"的内容')
    parser.add_argument('--response ', type=dict, default=[], help='')
    parser.add_argument('--state', type=dict, default=[], help='所有内容')
    parser.add_argument('--reward_scale_rate', type=int, default=20, help='奖励范围')
    parser.add_argument('--gsm_dataset', type=str, default='/home/lihaoran/lhr/maq/saved_data/gsm/gsm_3_2_20240319141252.json', help='处理数据集：包括correct、question、response(三个智能体的回答')

    args = parser.parse_args()
    return args


class RandomEnv(Env):
    # 定义一个 RandomEnv 类，继承自 Env 类

    def __init__(self, args):
        super(RandomEnv, self).__init__()
        self.args = args
        self.action_space = args.action_space
        self.obs_space = args.obs_space
        self.state_space = args.state_space
        self.num_agent = args.num_agents

        self.observation = args.observation
        self.state = args.state
        self.standard_answer = args.standard_answer
        self.response = args.response

        self.reward_fun = RewardFunction(self.standard_answer, args.reward_scale_rate)


        self.reset()

    def reset(self):
        self.max_episode_steps = np.random.randint(0, self.args.max_episode_steps, 1)[0]
        self.current_step = 0
        self.done = False

        self.state = self.args.state
        self.observation = self.args.observation

        return self.observation, self.state

    def step(self, actions):
        assert len(actions) == self.num_agent

        if self.current_step >= self.max_episode_steps:
            self.done = True

        self.current_step += 1
        self.state = self.args.state
        self.observation = self.args.observation
        self.response = self.args.response

        self.rewards = self.reward_fun.calculate_consistency_reward(self.response, self.standard_answer)

        return self.rewards, self.done, []

    def get_obs(self):
        # print(self.observation)
        return self.observation

    def get_state(self):
        return self.state

    def get_standard_answer(self):
        return self.standard_answer


