import torch
from torch import nn

def mlp(input_size, hidden_sizes=(64, 64), activation='tanh'):

    if activation == 'tanh':
        activation = nn.Tanh
    elif activation == 'relu':
        activation = nn.ReLU
    elif activation == 'sigmoid':
        activation = nn.Sigmoid

    layers = []
    sizes = (input_size, ) + hidden_sizes
    for i in range(len(hidden_sizes)):
        layers += [nn.Linear(sizes[i], sizes[i+1]), activation()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation='tanh', log_std=-0.5):
        super().__init__()

        self.mlp_net = mlp(state_dim, hidden_sizes, activation)
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.logstd_layer = nn.Parameter(torch.ones(1, action_dim) * log_std)
        self.mean_layer.weight.data.mul_(0.1)
        self.mean_layer.bias.data.mul_(0.0)

    def forward(self, x):
        out = self.mlp_net(x)
        mu = self.mean_layer(out)
        if len(mu.size()) == 1:
            mu = mu.view(1, -1)
        logstd = self.logstd_layer.expand_as(mu)
        std = torch.exp(logstd)
        return mu, std


    def take_action(self, state):
        mu, std = self.forward(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return action.detach().cpu().numpy().squeeze()


    def log_prob(self, states, actions, old):
        mu, std = self.forward(states)
        if old:
            action_dists = torch.distributions.Normal(mu.detach(), std.detach())
            return action_dists.log_prob(actions).sum(-1, keepdim=True), mu.detach(), std.detach()
        else:
            action_dists = torch.distributions.Normal(mu, std)
            return action_dists.log_prob(actions).sum(-1, keepdim=True), mu, std



class Value(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(64, 64), activation='tanh'):
        super().__init__()

        self.mlp_net = mlp(state_dim, hidden_sizes, activation)
        self.v_head = nn.Linear(hidden_sizes[-1], 1)

        self.v_head.weight.data.mul_(0.1)
        self.v_head.bias.data.mul_(0.0)

    def forward(self, obs):
        mlp_out = self.mlp_net(obs)
        v_out = self.v_head(mlp_out)
        return v_out