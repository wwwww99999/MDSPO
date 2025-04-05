import random
import torch
from train import train

if __name__ == '__main__':
    env_name = 'SafetyCarGoal2-v0'
    batch_size = 20000
    gae_r_lambda = 0.95
    gae_c_lambda = 0.95
    r_gamma = 0.99
    c_gamma = 0.99
    max_iter_num = 500
    epoch = 40
    minibatch = 64
    policy_lr = 3e-4
    r_value_lr = 3e-4
    c_value_lr = 3e-4
    l2_reg = 1e-3
    hidden_dim = 64
    eta_kl = 1
    target_kl = 0.01
    lam = 0.001
    lam_max = 2
    lam_lr = 3.5
    cost_threshold = 25
    activation = 'tanh'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in random.sample(range(1, 100), 5):
        train(env_name, batch_size, gae_r_lambda, gae_c_lambda, r_gamma, c_gamma, max_iter_num,
             epoch, minibatch, policy_lr, r_value_lr, c_value_lr, l2_reg, hidden_dim, eta_kl,
             target_kl, lam, lam_max, lam_lr, cost_threshold, seed, activation, device)
