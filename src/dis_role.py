#coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import torch
from torch.distributions import kl_divergence
import torch.distributions as D
import math
import time


class LatentCEDisRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentCEDisRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.num_agents
        self.n_actions = args.num_actions
        self.latent_dim = 256
        self.hidden_dim = args.rnn_hidden_dim
        self.bs = 1

        self.embed_fc_input_size = args.obs_space
        NN_HIDDEN_SIZE = 64
        activation_func = nn.LeakyReLU()

        self.embed_net = nn.Sequential(nn.Linear(self.embed_fc_input_size, NN_HIDDEN_SIZE),
                                       nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                       activation_func,
                                       nn.Linear(NN_HIDDEN_SIZE, self.latent_dim * 2))

        self.inference_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim + input_shape, NN_HIDDEN_SIZE),
                                           nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                           activation_func,
                                           nn.Linear(NN_HIDDEN_SIZE, self.latent_dim * 2))

        self.latent = th.rand(self.n_agents, self.latent_dim * 2)
        self.latent_infer = th.rand(self.n_agents, self.latent_dim * 2)

        self.latent_net = nn.Sequential(nn.Linear(self.latent_dim, NN_HIDDEN_SIZE),
                                        nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                        activation_func)

        # Dis Net
        self.dis_net = nn.Sequential(nn.Linear(self.latent_dim * 2, NN_HIDDEN_SIZE ),
                                     nn.BatchNorm1d(NN_HIDDEN_SIZE ),
                                     activation_func,
                                     nn.Linear(NN_HIDDEN_SIZE , 1))

        self.mi= th.rand(self.n_agents*self.n_agents)
        self.dissimilarity = th.rand(self.n_agents*self.n_agents)
        self.kl_loss_weight= 0.0001
        self.h_loss_weight =  0.0001
        self.var_floor=0.002
        self.soft_constraint_weight = 1.0


    def forward(self, inputs, hidden_state, train_mode=False):
        inputs['o'] = torch.tensor(inputs['o'], dtype=torch.float)
        inputs = inputs['o'].reshape(-1, self.input_shape)
        h_in = hidden_state.reshape(-1, self.hidden_dim)

        embed_fc_input = inputs[:, - self.embed_fc_input_size:]

        self.latent = self.embed_net(embed_fc_input)
        self.latent[:, -self.latent_dim:] = th.clamp(th.exp(self.latent[:, -self.latent_dim:]), min=self.var_floor)

        latent_embed = self.latent.reshape(self.bs * self.n_agents, self.latent_dim * 2)

        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2))
        latent = gaussian_embed.rsample()

        if True:

            self.latent_infer = self.inference_net(th.cat([h_in.detach(), inputs], dim=1))
            self.latent_infer[:, -self.latent_dim:] = th.clamp(th.exp(self.latent_infer[:, -self.latent_dim:]),min=self.var_floor)
            gaussian_infer = D.Normal(self.latent_infer[:, :self.latent_dim], (self.latent_infer[:, self.latent_dim:]) ** (1 / 2))

            loss = gaussian_embed.entropy().sum(dim=-1).mean() * self.h_loss_weight + kl_divergence(gaussian_embed, gaussian_infer).sum(dim=-1).mean() * self.kl_loss_weight   # CE = H + KL
            loss = th.clamp(loss, max=2e3)
            ce_loss = th.log(1 + th.exp(loss))

            dis_loss = 0
            dissimilarity_cat = None
            mi_cat = None
            latent_dis = latent.clone().view(self.bs, self.n_agents, -1)
            latent_move = latent.clone().view(self.bs, self.n_agents, -1)
            for agent_i in range(self.n_agents):
                latent_move = th.cat(
                    [latent_move[:, -1, :].unsqueeze(1), latent_move[:, :-1, :]], dim=1)
                latent_dis_pair = th.cat([latent_dis[:, :, :self.latent_dim],
                                          latent_move[:, :, :self.latent_dim],
                                          ], dim=2)
                mi = th.clamp(gaussian_embed.log_prob(latent_move.view(self.bs * self.n_agents, -1)) + 13.9,
                              min=-13.9).sum(dim=1, keepdim=True) / self.latent_dim

                dissimilarity = th.abs(self.dis_net(latent_dis_pair.view(-1, 2 * self.latent_dim)))

                if dissimilarity_cat is None:
                    dissimilarity_cat = dissimilarity.view(self.bs, -1).clone()
                else:
                    dissimilarity_cat = th.cat([dissimilarity_cat, dissimilarity.view(self.bs, -1)], dim=1)
                if mi_cat is None:
                    mi_cat = mi.view(self.bs, -1).clone()
                else:
                    mi_cat = th.cat([mi_cat, mi.view(self.bs, -1)], dim=1)

            mi_min = mi_cat.min(dim=1, keepdim=True)[0]
            mi_max = mi_cat.max(dim=1, keepdim=True)[0]
            di_min = dissimilarity_cat.min(dim=1, keepdim=True)[0]
            di_max = dissimilarity_cat.max(dim=1, keepdim=True)[0]

            mi_cat = (mi_cat - mi_min) / (mi_max - mi_min + 1e-12)
            dissimilarity_cat = (dissimilarity_cat - di_min) / (di_max - di_min + 1e-12)

            dis_loss = - th.clamp(mi_cat + dissimilarity_cat, max=1.0).sum() / self.bs / self.n_agents
            dis_norm = th.norm(dissimilarity_cat, p=1, dim=1).sum() / self.bs / self.n_agents

            c_dis_loss = (dis_norm + self.soft_constraint_weight * dis_loss) / self.n_agents
            loss = ce_loss + c_dis_loss

            self.mi = mi_cat[0]
            self.dissimilarity = dissimilarity_cat[0]
        return loss, c_dis_loss, ce_loss
