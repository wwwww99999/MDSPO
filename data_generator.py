from statistics import mean
import numpy as np
import torch
from environment import get_velocity_threshold

class DataGenerator:
    def __init__(self, state_dim, action_dim, batch_size, max_episode_steps, env, env_name,
                 gae_r_lambda, gae_c_lambda, r_gamma, c_gamma, normalize, seed, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.max_episode_steps = max_episode_steps
        self.env = env
        self.env_name = env_name
        self.gae_r_lambda = gae_r_lambda
        self.gae_c_lambda = gae_c_lambda
        self.r_gamma = r_gamma
        self.c_gamma = c_gamma
        self.normalize = normalize
        self.seed = seed
        self.device = device

        # Batch buffer
        self.state_buf = np.zeros((batch_size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((batch_size, action_dim),  dtype=np.float32)
        self.adv_r_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.adv_c_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.td_r_target_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.td_c_target_buf = np.zeros((batch_size, 1), dtype=np.float32)

        # Episode buffer
        self.state_eps = np.zeros((max_episode_steps, state_dim),  dtype=np.float32)
        self.next_state_eps = np.zeros((max_episode_steps, state_dim),  dtype=np.float32)
        self.action_eps = np.zeros((max_episode_steps, action_dim),  dtype=np.float32)
        self.reward_eps = np.zeros((max_episode_steps, 1),  dtype=np.float32)
        self.cost_eps = np.zeros((max_episode_steps, 1), dtype=np.float32)
        self.done_eps = np.zeros((max_episode_steps, 1), dtype=np.bool8)

        self.episode_steps = 0

    def interact(self, policy, r_value, c_value):
        batch_idx = 0
        average_episode_r = []
        average_episode_c = []
        while batch_idx < self.batch_size:
            state = self.env.reset()[0]
            state = self.normalize.normalize(state)
            r_eps = 0
            c_eps = 0
            for t in range(self.max_episode_steps):
                action = policy.take_action(torch.tensor(state, dtype=torch.float).to(self.device))
                next_state, r, c, terminated, truncated, info = self.env.step(action)
                if 'Velocity' in self.env_name:
                    v_threshold = get_velocity_threshold(self.env_name)
                    if 'y_velocity' not in info:
                        velocity = np.abs(info['x_velocity'])
                    else:
                        velocity = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
                    if velocity > v_threshold:
                        c = 1
                    else:
                        c = 0
                r_eps += r
                c_eps += c
                next_state = self.normalize.normalize(next_state)

                # Store in episode buffer
                self.state_eps[t] = state
                self.action_eps[t] = action
                self.next_state_eps[t] = next_state
                self.reward_eps[t] = r
                self.cost_eps[t] = c
                self.done_eps[t] = terminated
                state = next_state
                batch_idx += 1
                self.episode_steps += 1

                if terminated or truncated:
                    average_episode_r.append(r_eps)
                    average_episode_c.append(c_eps)
                    break

                if batch_idx == self.batch_size:
                    break

            self.state_eps = self.state_eps[:self.episode_steps]
            self.action_eps = self.action_eps[:self.episode_steps]
            self.next_state_eps = self.next_state_eps[:self.episode_steps]
            self.reward_eps = self.reward_eps[:self.episode_steps]
            self.cost_eps = self.cost_eps[:self.episode_steps]
            self.done_eps = self.done_eps[:self.episode_steps]

            adv_r, adv_c, td_r_target, td_c_target = self.GAE(r_value, c_value)

            # Update batch buffer
            start_idx = batch_idx - self.episode_steps
            end_idx = batch_idx

            self.state_buf[start_idx: end_idx] = self.state_eps
            self.action_buf[start_idx: end_idx] = self.action_eps
            self.adv_r_buf[start_idx: end_idx] = adv_r
            self.adv_c_buf[start_idx: end_idx] = adv_c
            self.td_r_target_buf[start_idx: end_idx] = td_r_target
            self.td_c_target_buf[start_idx: end_idx] = td_c_target

            # Reset episode buffer and update pointer
            self.state_eps = np.zeros((self.max_episode_steps, self.state_dim), dtype=np.float32)
            self.next_state_eps = np.zeros((self.max_episode_steps, self.state_dim), dtype=np.float32)
            self.action_eps = np.zeros((self.max_episode_steps, self.action_dim), dtype=np.float32)
            self.reward_eps = np.zeros((self.max_episode_steps, 1), dtype=np.float32)
            self.cost_eps = np.zeros((self.max_episode_steps, 1), dtype=np.float32)
            self.done_eps = np.zeros((self.max_episode_steps, 1), dtype=np.bool8)
            self.episode_steps = 0

            # Normalize advantage functions
        self.adv_r_buf = (self.adv_r_buf - self.adv_r_buf.mean()) / (self.adv_r_buf.std() + 1e-6)
        self.adv_c_buf = (self.adv_c_buf - self.adv_c_buf.mean()) / (self.adv_c_buf.std() + 1e-6)

        return {'states': self.state_buf, 'actions': self.action_buf,
                'r_advantages': self.adv_r_buf, 'c_advantages': self.adv_c_buf,
                'td_r_target': self.td_r_target_buf, 'td_c_target': self.td_c_target_buf,
                'average_episode_r': mean(average_episode_r),
                'average_episode_c': mean(average_episode_c)}

    def GAE(self, r_value, c_value):
        adv_r = np.zeros((self.episode_steps, 1))
        prev_adv_r = 0
        adv_c = np.zeros((self.episode_steps, 1))
        prev_adv_c = 0
        v_r_t = r_value(torch.tensor(self.state_eps, dtype=torch.float).to(self.device)).detach().cpu().numpy()
        v_r_tplus1 = r_value(torch.tensor(self.next_state_eps, dtype=torch.float).to(self.device)).detach().cpu().numpy()
        v_c_t = c_value(torch.tensor(self.state_eps, dtype=torch.float).to(self.device)).detach().cpu().numpy()
        v_c_tplus1 = c_value(torch.tensor(self.next_state_eps, dtype=torch.float).to(self.device)).detach().cpu().numpy()
        td_r_delta = self.reward_eps + self.r_gamma * v_r_tplus1 * (1 - self.done_eps) - v_r_t
        td_c_delta = self.cost_eps + self.c_gamma * v_c_tplus1 * (1 - self.done_eps) - v_c_t
        for t in reversed(range(self.episode_steps)):
            adv_r[t] = td_r_delta[t] + self.r_gamma * self.gae_r_lambda * prev_adv_r
            adv_c[t] = td_c_delta[t] + self.c_gamma * self.gae_c_lambda * prev_adv_c
            prev_adv_r = adv_r[t]
            prev_adv_c = adv_c[t]
        td_r_target = v_r_t + adv_r
        td_c_target = v_c_t + adv_c
        return adv_r, adv_c, td_r_target, td_c_target