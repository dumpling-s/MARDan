import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time


class RolloutWorker:

    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.num_actions = args.num_actions
        self.num_agents = args.num_agents
        self.state_space = args.state_space
        self.obs_space = args.obs_space
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon


    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded, a= [], [], [], [], [], [], [], [] ,[]
        self.env.reset()
        terminated = False

        step = 0
        episode_reward = np.zeros(3)
        self.agents.policy.init_hidden(1)



        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated:
            obs = self.env.get_obs()

            state = self.env.get_state()
            standard_answer = self.env.get_standard_answer()

            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.num_agents):
                avail_action = None
                action = self.agents.choose_action(obs[agent_id], agent_id, avail_action,
                                                   epsilon, evaluate)
                action = np.array(action[0].cpu())
                action_onehot = np.zeros(self.args.num_actions)
                # action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                a.append(standard_answer)


            # print(state)
            reward, terminated, _ = self.env.step(actions)
            if step == self.args.max_episode_steps - 1:
                terminated = 1

            o.append(obs)
            s.append(state)
            u.append(actions)
            u_onehot.append(actions_onehot)
            r.append(reward)
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            a.append(standard_answer)
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        a = a[:-1]

        for i in range(step, self.args.max_episode_steps):
            o.append(np.zeros((self.num_agents, self.obs_space)))
            u.append(np.zeros([self.num_agents, self.args.num_actions]))
            # u.append([np.zeros((self.args.num_actions,self.args.action_space)),np.zeros((self.args.num_actions,self.args.action_space),np.zeros((self.args.num_actions,self.args.action_space)])
            s.append(np.zeros(self.state_space))
            r.append(np.zeros(3))
            o_next.append(np.zeros((self.num_agents, self.obs_space)))
            u_onehot.append(np.zeros((self.num_agents, self.num_actions)))
            s_next.append(np.zeros(self.state_space))
            padded.append([1.])
            terminate.append([1.])
            a.append(np.zeros((self.num_agents, self.num_actions)))

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                           u_onehot=u_onehot.copy(),
                           padded=padded.copy(),
                           terminated=terminate.copy(),
                           a=a.copy()
                           )

        for key in episode.keys():
            try:
                episode[key] = np.array([episode[key]])
            except Exception as e:
                    print(key)
                    print(episode[key])
                    print("发生了一个错误：", e)

        if not evaluate:
            self.epsilon = epsilon

        return episode, episode_reward


