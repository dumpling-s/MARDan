import torch.nn as nn
import torch
import torch.nn.functional as f


class AgentWithLLMAndRNN(nn.Module):
    def __init__(self, llm_model, args, device='cuda'):
        super(AgentWithLLMAndRNN, self).__init__()
        self.device = device
        self.args = args

        self.llm = llm_model.to(device)  # 接收外部传入的 LLM
        self.llm.train()

        self.embedding_dim = self.llm.config.hidden_size

        self.rnn = nn.GRU(self.embedding_dim, self.args.rnn_hidden_dim, batch_first=True)
        self.q_head = nn.Linear(self.args.rnn_hidden_dim, self.args.num_actions)

    def forward(self, input_ids, attention_mask, hidden_state):
        outputs = self.llm(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # [B, T, D]
        pooled = last_hidden[:, 0, :]  # [B, D]

        rnn_output, next_hidden = self.rnn(pooled.to(dtype=torch.float32), hidden_state.unsqueeze(0))   # [B, 1, H], [1, B, H]
        q_value = self.q_head(rnn_output.squeeze(1))  # [B, num_actions]

        return q_value, next_hidden
        
class RNN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        # print(input_shape)
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.num_actions)

    def forward(self, obs, hidden_state):
        # 前向传播函数，接收观察值和隐藏状态，返回动作值和新的隐藏状态
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)  # 隐藏状态
        q = self.fc2(h)  # 动作值
        return q, h


class RNN_action(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN_action, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_space)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h = self.rnn(x, hidden_state)
        q = self.fc2(h).view(150, 32000)
        # print(q)
        return q, h
class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args

        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.state_space, args.num_agents * args.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(args.state_space, args.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(args.state_space, args.qmix_hidden_dim)
        self.hyper_b2 =nn.Sequential(nn.Linear(args.state_space, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1))

    def forward(self, q_values, states):
        episode_num = q_values.size(0)
        q_values = q_values.reshape(-1,  self.args.num_actions, self.args.num_agents)
        states = states.reshape(-1, self.args.state_space)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, self.args.num_agents, self.args.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)

        hidden = f.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total
