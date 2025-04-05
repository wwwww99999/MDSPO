import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, kl

class Agent:
    def __init__(self, policy, r_value, c_value, epoch, minibatch, policy_lr, r_value_lr, c_value_lr,
                 l2_reg, eta_kl, target_kl, c_gamma, lam, lam_max, lam_lr, cost_threshold, device):
        self.policy = policy
        self.r_value = r_value
        self.c_value = c_value
        self.epoch = epoch
        self.minibatch = minibatch
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.r_value_optimizer = torch.optim.Adam(self.r_value.parameters(), lr=r_value_lr)
        self.c_value_optimizer = torch.optim.Adam(self.c_value.parameters(), lr=c_value_lr)
        self.l2_reg = l2_reg
        self.eta_kl = eta_kl
        self.target_kl = target_kl
        self.c_gamma = c_gamma
        self.lam = lam
        self.lam_max = lam_max
        self.lam_lr = lam_lr
        self.cost_threshold = cost_threshold
        self.device = device

    def update(self, data):
        states = torch.tensor(data['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(data['actions'], dtype=torch.float).to(self.device)
        r_adv = torch.tensor(data['r_advantages'], dtype=torch.float).to(self.device)
        c_adv = torch.tensor(data['c_advantages'], dtype=torch.float).to(self.device)
        td_r_target = torch.tensor(data['td_r_target'], dtype=torch.float).to(self.device)
        td_c_target = torch.tensor(data['td_c_target'], dtype=torch.float).to(self.device)

        old_log_prob_k, old_mu_k, old_std_k = self.policy.log_prob(states, actions, True)

        dataset_1 = TensorDataset(states, actions, r_adv, old_log_prob_k, old_mu_k, old_std_k)
        loader_1 = DataLoader(dataset=dataset_1, batch_size=self.minibatch, shuffle=True)

        #phase 1
        for i in range(self.epoch):
            for _, (states_b, actions_b, r_adv_b, old_log_prob_k_b, old_mu_k_b, old_std_k_b) in enumerate(loader_1):

                # Update policy
                log_prob, mu, std = self.policy.log_prob(states_b, actions_b, False)
                kl_divergence_1 = gaussian_kl(mu, std, old_mu_k_b, old_std_k_b)
                ratio = torch.exp(log_prob - old_log_prob_k_b)
                policy_loss = torch.mean(-ratio * r_adv_b + kl_divergence_1 / self.eta_kl)
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.policy_optimizer.step()

            if exceed_kl_bound(self.policy, states, actions, old_mu_k, old_std_k, self.target_kl):
                break

        #phase 2 and 3
        j_c = data['average_episode_c']
        old_log_prob_k_0_5, old_mu_k_0_5, old_std_k_0_5 = self.policy.log_prob(states, actions, True)

        self.lam += self.lam_lr * (torch.mean(torch.exp(old_log_prob_k_0_5 - old_log_prob_k) * c_adv).item() +
                                   (1 - self.c_gamma) * (j_c - self.cost_threshold))
        # self.lam += self.lam_lr * (j_c - self.cost_threshold)
        self.lam = max(self.lam, 0)
        self.lam = min(self.lam, self.lam_max)
        # print(self.lam)

        dataset_2 = TensorDataset(states, actions, c_adv,
                                  old_log_prob_k, old_mu_k_0_5, old_std_k_0_5)
        loader_2 = DataLoader(dataset=dataset_2, batch_size=self.minibatch, shuffle=True)

        for i in range(self.epoch):
            for _, (states_b, actions_b, c_adv_b, old_log_prob_k_b,
                    old_mu_k_0_5_b, old_std_k_0_5_b) in enumerate(loader_2):

                # Update policy
                log_prob, mu, std = self.policy.log_prob(states_b, actions_b, False)
                kl_divergence_2 = gaussian_kl(mu, std, old_mu_k_0_5_b, old_std_k_0_5_b)
                ratio = torch.exp(log_prob - old_log_prob_k_b)
                policy_loss = torch.mean(kl_divergence_2 + self.lam * ratio * c_adv_b)
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.policy_optimizer.step()

            if exceed_kl_bound(self.policy, states, actions, old_mu_k_0_5, old_std_k_0_5, self.target_kl):
                break


        dataset_3 = TensorDataset(states, td_r_target, td_c_target)
        loader_3 = DataLoader(dataset=dataset_3, batch_size=self.minibatch, shuffle=True)

        for _, (states_b, td_r_target_b, td_c_target_b) in enumerate(loader_3):

            # Update reward critic
            r_value_loss = torch.mean(F.mse_loss(self.r_value(states_b), td_r_target_b))
            for param in self.r_value.parameters():
                r_value_loss += param.pow(2).sum() * self.l2_reg
            self.r_value_optimizer.zero_grad()
            r_value_loss.backward()
            self.r_value_optimizer.step()

            # Update cost critic
            c_value_loss = torch.mean(F.mse_loss(self.c_value(states_b), td_c_target_b))
            for param in self.c_value.parameters():
                c_value_loss += param.pow(2).sum() * self.l2_reg
            self.c_value_optimizer.zero_grad()
            c_value_loss.backward()
            self.c_value_optimizer.step()

def gaussian_kl(mean1, std1, mean2, std2):
    normal1 = Normal(mean1, std1)
    normal2 = Normal(mean2, std2)
    return kl.kl_divergence(normal1, normal2).sum(-1, keepdim=True)


def exceed_kl_bound(policy, states, actions, old_mu, old_std, target_kl):
    _, mu, std = policy.log_prob(states, actions, True)
    if gaussian_kl(mu, std, old_mu, old_std).mean().item() > target_kl:
        return True
    else:
        return False