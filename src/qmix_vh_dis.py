import torch
import os
import torch.nn as nn
from qmix_net import AgentWithLLMAndRNN
from qmix_net import QMixNet
from qmix_net import RNN_action
from dis_role import LatentCEDisRNNAgent
import numpy as np
import datetime

class QMIX:
    def __init__(self, args):
        self.num_actions = args.num_actions
        self.num_agents = args.num_agents
        self.state_space = args.state_space
        self.obs_space = args.obs_space
        self.input_shape = self.obs_space
        if args.last_action:
            self.input_shape += self.num_actions
        if args.reuse_network:
            self.input_shape += self.num_agents

        self.dis_net = LatentCEDisRNNAgent(self.input_shape, args)
        self.eval_rnn = nn.ModuleList([
            AgentWithLLMAndRNN(args.model_lora[_], args) for _ in range(self.num_agents)
        ])
        self.target_rnn = nn.ModuleList([
            AgentWithLLMAndRNN(args.model_lora[_], args) for _ in range(self.num_agents)
        ])
        self.eval_qmix_net = QMixNet(args)
        self.target_qmix_net = QMixNet(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()


        self.model_dir = args.model_dir + '/' + args.alg
        if self.args.load_model_before:
            if os.path.exists(self.model_dir + str(self.args.save_cycle) + '/' + '_qmix_net_params.pkl'):
                for i, agent in enumerate(self.eval_rnn):
                    agent.load_state_dict(
                        torch.load(f'{self.model_dir + str(self.args.save_cycle)}/_agentNet{i}_params.pth'), weights_only=True)
                path_qmix = self.model_dir + str(self.args.save_cycle) + '/' + '_qmix_net_params.pkl'

                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, weights_only=True))
                print('Successfully load the model: {} and {}'.format(path_qmix))
            else:
                raise Exception("No model!")

        for i in range(self.num_agents):
            self.target_rnn[i].load_state_dict(self.eval_rnn[i].state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())


        self.eval_parameters = list(self.eval_qmix_net.parameters())
        for agent in self.eval_rnn:
            self.eval_parameters += list(agent.parameters())


        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        self.eval_hidden = None
        self.target_hidden = None
        self.evaluate_cycle = args.evaluate_cycle

    def learn(self, batch, train_step):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            # print(batch[key].dtype)
            if key == 's' or key == 's_next' or key == 'r':
                batch[key] = torch.tensor(batch[key], dtype=torch.float16)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.long)

        s, s_next, u, r, avail_u, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                              batch['r'],  batch['avail_u'], batch['terminated']
        mask = 1 - batch["padded"]

        role_loss, role_c_dis_loss, role_ce_loss = self.dis_net.forward(batch, torch.zeros((episode_num, self.num_agents, self.args.rnn_hidden_dim)))
        q_evals, q_targets = self.get_q_values(batch)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
            
        q_evals = q_evals.clone().detach().squeeze(1).type(torch.float32))
        # s = torch.tensor(s, dtype=torch.float32)
        s = s.clone().detach().type(torch.float32)
        q_targets = q_targets.squeeze(1)
        s_next = s_next.clone().detach().requires_grad_(True).type(torch.float32)
        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)
        targets = r + self.args.gamma * q_total_target

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error


        loss = (masked_td_error ** 2).sum() / mask.sum() - self.args.gamma * role_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        print(loss)

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            for i in range(self.num_agents):
                self.target_rnn[i].load_state_dict(self.eval_rnn[i].state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        # torch.cuda.empty_cache()

    def _get_inputs(self, batch):
        obs, a,  u_onehot = batch['o'], batch['a'], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_a = [], []
        inputs.append(obs)  # torch.Size([1, 1, 3, 256])
        # inputs_next.append(obs_next)
        inputs_a.append(a)

        if self.args.last_action:
            inputs.append(torch.zeros_like(u_onehot))
            # inputs_next.append(torch.zeros_like(u_onehot))
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.num_agents).unsqueeze(0).expand(episode_num, -1, -1))
            # inputs_next.append(torch.eye(self.args.num_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num, self.args.num_agents, -1) for x in inputs], dim=1)
        # inputs_next = torch.cat([x.reshape(episode_num, self.args.num_agents, -1) for x in inputs_next], dim=1)
        inputs_a =torch.cat([x.reshape(episode_num, self.args.num_agents, -1) for x in inputs_a], dim=1)
        aa = inputs.shape[0]

        return inputs, inputs_a, aa

    def get_q_values(self, batch):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        inputs, inputs_a, aa = self._get_inputs(batch)
        # print(inputs.shape)
        if self.args.cuda:
            inputs = inputs.cuda()
            inputs_a = inputs_a.cuda()
            self.eval_hidden = self.eval_hidden.cuda()
            self.target_hidden = self.target_hidden.cuda()

        attention_mask = torch.tensor([1] * len(inputs)).unsqueeze(0).cuda().to(torch.long)
        # pad_token_id = tokenizer.eos_token_id

        q_values_list = []
        q_targets_list = []
        for e_n in range(episode_num):
            for agent_id in range(self.num_agents):
                eval_agent = self.eval_rnn[agent_id]
                target_agent = self.target_rnn[agent_id]
                eval_input = inputs[e_n][agent_id]
                target_input = inputs_a[e_n][agent_id]
                eval_hidden = self.eval_hidden[e_n][agent_id]
                target_hidden = self.target_hidden[e_n][agent_id]

                eval_q_values, eval_next_hidden = eval_agent(eval_input, attention_mask, eval_hidden)
                target_q_values, target_next_hidden = target_agent(target_input, attention_mask, target_hidden)
                q_values_list.append(eval_q_values)
                q_targets_list.append(target_q_values)
        q_eval = torch.stack(q_values_list)
        q_target = torch.stack(q_targets_list)
        # q_eval = value_e
        # q_target = value_t
        q_eval = q_eval.view(episode_num, self.num_agents, -1)
        q_target = q_target.view(episode_num, self.num_agents, -1)
        q_evals.append(q_eval)
        q_targets.append(q_target) 


        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.num_agents, self.args.rnn_hidden_dim), dtype=torch.float)
        self.target_hidden = torch.zeros((episode_num, self.num_agents, self.args.rnn_hidden_dim), dtype=torch.float)

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        for i, agent in enumerate(self.eval_rnn):
            torch.save(agent.state_dict(), f'{self.model_dir+str(self.args.save_cycle)}/{num}_agentNet{i}_params.pth')
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + str(self.args.save_cycle)+ '/' + num + '_qmix_net_params.pkl')
