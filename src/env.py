from gym.core import Env
import argparse
import numpy as np
from utils import RewardFunction
import torch.nn as nn
from tqdm import tqdm
import torch

def qmix_args(args):
    args.rnn_hidden_dim = 64
    args.two_hyper_layers = False
    args.qmix_hidden_dim = 32
    args.lr = 1e-7

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch to train the agent
    args.n_epoch = 50

    # the number of the episodes in one epoch
    args.n_episodes = 2

    # the number of the train steps in one epoch
    args.train_steps = 1

    # # how often to evaluate
    args.evaluate_cycle = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = 100

    # how often to save the model
    args.save_cycle = 1

    # how often to update the target_net
    args.target_update_cycle = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10



    return args


def get_common_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--obs_space', type=int, default=2560, help='observation space')


    parser.add_argument('--state_space', type=int, default=2560, help='state_space')


    parser.add_argument('--action_space', type=int, default=32000, help='')
    # parser.add_argument('--num_actions', type=int, default=1, help='')
    parser.add_argument('--num_actions', type=int, default=2560, help='')
    parser.add_argument('--num_agents', type=int, default=3, help='number of agents')
    parser.add_argument('--round', type=int, help='辩论轮次')
    parser.add_argument('--max_episode_steps', type=int, default=1, help='')

    # parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    # parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    # parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=2, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='the directory of save the replay')

    parser.add_argument('--alg', type=str, default='qmix_', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=False, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=False, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--n_evaluate_episode', type=int, default=3, help='number of the episode to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='', help='result directory of the policy')

    parser.add_argument('--load_model', type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--load_model_before', type=bool, default=False, help='whether to load the pretrained model')

    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--threshold', type=float, default=19.9, help='threshold to judge whether win')


    parser.add_argument('--standard_answer', type=list, default=[], help='标准答案')
    parser.add_argument('--observation', type=list, default=[], help='观测值即"role": "user", "content"的内容')
    parser.add_argument('--response', type=list, default=[], help='智能体的回答')
    parser.add_argument('--state', type=list, default=[], help='')
    parser.add_argument('--model_lora', type=nn.ModuleList(), default=[], help='')

    args = parser.parse_args()
    return args


class RandomEnv(Env):
    def __init__(self, args):
        super(RandomEnv, self).__init__()  # 调用父类的初始化方法
        self.args = args  # 保存传入的参数
        self.action_space = args.action_space  # 设置动作空间
        self.obs_space = args.obs_space  # 设置观察空间
        self.state_space = args.state_space  # 设置状态空间
        self.num_agent = args.num_agents  # 设置智能体数量

        self.observation = args.observation  # 初始化观察变量
        self.state = args.state  # 初始化状态变量
        self.standard_answer = args.standard_answer
        self.response = args.response
        self.round = args.round
        self.reward_fun = RewardFunction(self.standard_answer, args.reward_scale_rate)


        self.reset()

    def reset(self):
        self.max_episode_steps = np.random.randint(0, self.args.max_episode_steps, 1)[0]
        self.current_step = 0
        self.done = False
        self.state = self.args.state
        self.observation = self.args.observation
        self.standard_answer = self.args.standard_answer
        self.response = self.args.response

    def step(self, actions):
        assert len(actions) == self.num_agent

        if self.current_step >= self.max_episode_steps:
            self.done = True

        self.current_step += 1
        self.state = self.args.state
        self.observation = self.args.observation
        self.standard_answer = self.args.standard_answer
        self.rewards = self.reward_fun.calculate_consistency_reward(actions, self.standard_answer)
        return self.rewards, self.done, []

    def get_obs(self):
        # print(self.observation)
        return self.observation  # 返回观察值

    def get_state(self):
        return self.state  # 返回状态值

    def get_standard_answer(self):
        return self.standard_answer

    def get_response(self):
        return self.response


