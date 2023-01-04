import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import replay_memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
class SACAgent():
    def __init__(
        self,
        n_actions,
        input_shape,
        model,
        device,
        learning_rate = [1e-4, 2e-4],
        reward_decay = 0.98,
        memory_size = 10000,
        batch_size = 32,
        tau = 0.01,
        alpha = 0.5,
        auto_entropy_tuning = True,
    ):
        # initialize parameters
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = device
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.learn_step_counter = 0
        self._build_net(model[0], model[1])
        self.memory = replay_memory.Memory(memory_size)
        self.entropy = 0

        if self.auto_entropy_tuning == True:
            self.target_entropy = -torch.Tensor(self.n_actions).to(device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=0.0002)

    def _build_net(self, anet, cnet):
        # Policy Network
        self.actor = anet(self.input_shape, self.n_actions).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        
        # Q Network
        self.critic = cnet(self.input_shape, self.n_actions).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
        self.critic_target = cnet(self.input_shape, self.n_actions).to(self.device)
        self.critic_target.eval()
        
    def choose_action(self, s, eval=False):
        b_s = {}
        for key in s:
            b_s[key] = torch.FloatTensor(np.expand_dims(s[key],0)).to(self.device)
        if eval == False:
            action, _, _ = self.actor.sample(b_s)
        else:
            _, _, action = self.actor.sample(b_s)
        action = action.cpu().detach().numpy()[0]
        return action

    def store_transition(self, s, a, r, s_, d):
        self.memory.push(s, a, r, s_, d)

    def soft_update(self, TAU=0.01):
        with torch.no_grad():
            for targetParam, evalParam in zip(self.critic_target.parameters(), self.critic.parameters()):
                targetParam.copy_((1 - self.tau)*targetParam.data + self.tau*evalParam.data)

    def learn(self):
        # sample batch memory from all memory
        b_s, b_a, b_r, b_s_, b_d = self.memory.sample_torch(self.batch_size, self.device, False) 
        
        with torch.no_grad():
            a_next, logpi_next, _ = self.actor.sample(b_s_)
            q_next_target = self.critic_target(b_s, a_next) - self.alpha * logpi_next
            q_target = b_r + (1-b_d) * self.gamma * q_next_target
        
        q_eval = self.critic(b_s, b_a)
        self.critic_loss = nn.MSELoss()(q_eval, q_target)
        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()

        a_curr, logpi_curr, _ = self.actor.sample(b_s)
        q_current = self.critic(b_s, a_curr)
        self.actor_loss = ((self.alpha*logpi_curr) - q_current).mean()
        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()
        
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logpi_curr + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp().detach().cpu().numpy())
        
        # increasing epsilon
        self.learn_step_counter += 1
        return float(self.actor_loss.detach().cpu().numpy()), float(self.critic_loss.detach().cpu().numpy())

    def save_model(self, path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        anet_path = os.path.join(path, "sac_anet.pt")
        cnet_path = os.path.join(path, "sac_cnet.pt")
        torch.save(self.critic.state_dict(), cnet_path)
        torch.save(self.actor.state_dict(), anet_path)

    def load_model(self, path):
        import os
        anet_path = os.path.join(path, "sac_anet.pt")
        cnet_path = os.path.join(path, "sac_cnet.pt")
        self.critic.load_state_dict(torch.load(cnet_path, map_location=self.device))
        self.critic_target.load_state_dict(torch.load(cnet_path, map_location=self.device))
        self.actor.load_state_dict(torch.load(anet_path, map_location=self.device))