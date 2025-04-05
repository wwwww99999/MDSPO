import time
from statistics import mean
from tqdm import tqdm
import safety_gymnasium
import torch
import numpy as np
from net import GaussianPolicy, Value
from agent import Agent
from data_generator import DataGenerator
import csv
from normalize import Normalize


def train(env_name, batch_size, gae_r_lambda, gae_c_lambda, r_gamma, c_gamma, max_iter_num,
          epoch, minibatch, policy_lr, r_value_lr, c_value_lr, l2_reg, hidden_dim,
          eta_kl, target_kl, lam, lam_max, lam_lr, cost_threshold, seed, activation, device):

    env = safety_gymnasium.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_episode_steps = env._max_episode_steps


    policy = GaussianPolicy(state_dim, action_dim, (hidden_dim, hidden_dim), activation).to(device)
    r_value = Value(state_dim, (hidden_dim, hidden_dim), activation).to(device)
    c_value = Value(state_dim, (hidden_dim, hidden_dim), activation).to(device)

    env.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    normalize = Normalize(clip=5)

    agent = Agent(policy, r_value, c_value, epoch, minibatch, policy_lr, r_value_lr, c_value_lr,
                  l2_reg, eta_kl, target_kl, c_gamma, lam, lam_max, lam_lr, cost_threshold, device)

    data_generator = DataGenerator(state_dim, action_dim, batch_size, max_episode_steps,
                                   env, env_name, gae_r_lambda, gae_c_lambda, r_gamma,
                                   c_gamma, normalize, seed, device)

    r = []
    c = []

    start = time.time()

    for j in range(10):
        with tqdm(total=int(max_iter_num / 10), desc='Iteration %d' % (j + 1)) as pbar:
            for i_episode in range(int(max_iter_num / 10)):
                interact_data = data_generator.interact(agent.policy, agent.r_value, agent.c_value)
                r.append(interact_data['average_episode_r'])
                c.append(interact_data['average_episode_c'])
                agent.update(interact_data)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (max_iter_num / 10 * j + i_episode + 1),
                                      'r_episode': '%.3f' % mean(r[-10:]),
                                      'c_episode': '%.3f' % mean(c[-10:])})
                pbar.update(1)

    end = time.time()
    print("Training Time:" '%.2f' %(end - start))

    with open("r_" + str(seed) + ".csv", "w", newline='') as file:
        # 创建一个 CSV writer 对象
        csv_writer = csv.writer(file)

        # 写入列表中的数据
        csv_writer.writerow(r)

    file.close()

    with open("c_" + str(seed) + ".csv", "w", newline='') as file:
        # 创建一个 CSV writer 对象
        csv_writer = csv.writer(file)

        # 写入列表中的数据
        csv_writer.writerow(c)

    file.close()

    torch.save(agent, 'agent')
