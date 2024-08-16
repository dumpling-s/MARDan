import torch
import os
# __package__ = '.src.qmix_33-master'
from qmix_net import RNN
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
        input_shape = self.obs_space
        if args.last_action:
            input_shape += self.num_actions
        if args.reuse_network:
            input_shape += self.num_agents

        self.dis_net = LatentCEDisRNNAgent(input_shape, args)

        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)
        self.RNN_action = RNN_action(input_shape, args)
        self.eval_qmix_net = QMixNet(args)
        self.target_qmix_net = QMixNet(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.model_dir + '/' + args.alg


        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        self.eval_hidden = None
        self.target_hidden = None
        self.evaluate_cycle = args.evaluate_cycle

        # print('Init QMIX')

    def learn(self, batch, train_step, model):
        if self.args.load_model_before:
            if train_step == 0:
                path_rnn = self.model_dir + 'xx' + '/' + '99_rnn_net_params.pkl'
                path_qmix = self.model_dir + 'xx' + '/' + '99_qmix_net_params.pkl'
                self.eval_rnn.load_state_dict(torch.load(path_rnn))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix))
                print('use exit model')
            if (train_step != 0) and ((train_step) % (self.evaluate_cycle) == 0):
                if os.path.exists(self.model_dir + str(self.args.save_cycle) + '/' + str(int(train_step / self.evaluate_cycle)-1) + '_rnn_net_params.pkl'):
                    path_rnn = self.model_dir + str(self.args.save_cycle) + '/' + str(int(train_step / self.evaluate_cycle)-1) + '_rnn_net_params.pkl'
                    path_qmix =self.model_dir + str(self.args.save_cycle)+ '/' + str(int(train_step / self.evaluate_cycle)-1) + '_qmix_net_params.pkl'
                    # print(path_rnn)
                    self.eval_rnn.load_state_dict(torch.load(path_rnn))
                    self.eval_qmix_net.load_state_dict(torch.load(path_qmix))
                    # print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
                else:
                    print(self.model_dir + str(self.args.save_cycle) + '/' + str(int(train_step / self.evaluate_cycle)-1) + '_rnn_net_params.pkl')
                    raise Exception("No model!")
        else:
            if ((train_step) % (self.evaluate_cycle) == 0) & (train_step != 0):
                if self.args.load_model:
                    if os.path.exists(self.model_dir + str(self.args.save_cycle) + '/' + str(
                            int(train_step / self.evaluate_cycle) - 1) + '_rnn_net_params.pkl'):
                        path_rnn = self.model_dir + str(self.args.save_cycle) + '/' + str(
                            int(train_step / self.evaluate_cycle) - 1) + '_rnn_net_params.pkl'
                        path_qmix = self.model_dir + str(self.args.save_cycle) + '/' + str(
                            int(train_step / self.evaluate_cycle) - 1) + '_qmix_net_params.pkl'
                        # print(path_rnn)
                        self.eval_rnn.load_state_dict(torch.load(path_rnn))
                        self.eval_qmix_net.load_state_dict(torch.load(path_qmix))
                        # print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
                    else:
                        print(self.model_dir + str(self.args.save_cycle) + '/' + str(
                            int(train_step / self.evaluate_cycle) - 1) + '_rnn_net_params.pkl')
                        raise Exception("No model!")

        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            # print(batch[key].dtype)
            if key == 's' or key == 's_next' or key == 'r':
                batch[key] = torch.tensor(batch[key], dtype=torch.float64)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.long)

        s, s_next, u, r, avail_u, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                              batch['r'],  batch['avail_u'], batch['terminated']
        mask = 1 - batch["padded"]
        # torch.Size([10,5,3,256) 应该是

        role_loss, role_c_dis_loss, role_ce_loss = self.dis_net.forward(batch, torch.zeros((episode_num, self.num_agents, self.args.rnn_hidden_dim)))
        # reg_loss /= batch.max_seq_length






        q_evals, q_targets = self.get_q_values(batch, model)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        q_evals = torch.tensor(q_evals, dtype=torch.float32).squeeze(1)
        s = torch.tensor(s, dtype=torch.float32)
        q_targets = q_targets.squeeze(1)
        s_next = torch.tensor(s_next, dtype=torch.float32)
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

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def _get_inputs(self, batch):
        obs, a,  u_onehot = batch['o'], batch['a'], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_a = [], []
        inputs.append(obs)
        inputs_a.append(a)

        if self.args.last_action:
            inputs.append(torch.zeros_like(u_onehot))
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.num_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num, self.args.num_agents, -1) for x in inputs], dim=1)
        inputs_a =torch.cat([x.reshape(episode_num, self.args.num_agents, -1) for x in inputs_a], dim=1)
        aa = inputs.shape[0]

        return inputs, inputs_a, aa

    def get_q_values(self, batch, model):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        inputs, inputs_a, aa = self._get_inputs(batch)
        # print(inputs.shape)
        if self.args.cuda:
            inputs = inputs.cuda()
            inputs_a = inputs_a.cuda()
        attention_mask = torch.tensor([1] * len(inputs)).unsqueeze(0).cuda().to(torch.long)
        # pad_token_id = tokenizer.eos_token_id
        model_WithValueHead = model
        with torch.no_grad():
            lm_logits_e, loss_e, value_e = model_WithValueHead.forward(input_ids=inputs[0].long(),
                                                  attention_mask=attention_mask,
                                                  )
            lm_logits_t, loss_t, value_t = model_WithValueHead.forward(input_ids=inputs_a[0].long(),
                                                  attention_mask=attention_mask,
                                                  )

        q_eval = value_e
        q_target = value_t
        q_eval = q_eval.view(episode_num, self.num_agents, -1)
        q_target = q_target.view(episode_num, self.num_agents, -1)
        q_evals.append(q_eval)  #
        q_targets.append(q_target)  #


        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.num_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.num_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + str(self.args.save_cycle)+ '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir + str(self.args.save_cycle)+ '/' + num + '_rnn_net_params.pkl')