import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import replay_memory

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
        target_entropy_ratio = 0.8#0.9,
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
        self._build_net(model[0], model[1])
        self.entropy = 0
        self.memory = replay_memory.Memory(memory_size)

        # Set trainable temperature.
        if self.auto_entropy_tuning == True:
            target_entropy = -np.log(1.0 / self.n_actions) * target_entropy_ratio
            self.target_entropy = torch.tensor(target_entropy).to(self.device) #self.n_actions
            # Because alpha is always positive, we optimize log_alpha instead.
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optim = optim.RMSprop([self.log_alpha], lr=0.0001)

    def _build_net(self, anet, cnet):
        # Policy Network
        self.actor = anet(self.input_shape, self.n_actions).to(self.device)
        #self.actor = anet(self.input_shape, self.n_actions, 32).to(self.device)
        #self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=self.lr[0], eps=0.001, alpha=0.95)
        
        # Evaluation Critic Network (new)
        self.critic = cnet(self.input_shape, self.n_actions).to(self.device)
        #self.critic = cnet(self.input_shape, self.n_actions, 32).to(self.device)
        #self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=self.lr[1], eps=0.001, alpha=0.95)
        
        # Target Critic Network (old)
        self.critic_target = cnet(self.input_shape, self.n_actions).to(self.device)
        #self.critic_target = cnet(self.input_shape, self.n_actions, 32).to(self.device)
        self.critic_target.eval()

    def choose_action(self, s, eval=False):
        b_s = {}
        for key in s:
            b_s[key] = torch.FloatTensor(np.expand_dims(s[key],0)).to(self.device)
        with torch.no_grad():
            action_dist = self.actor(b_s).view(-1).cpu().numpy()
        #print(action_dist)
        if eval:
            return np.argmax(action_dist)
        else:
            return np.random.choice(action_dist.shape[0], p=action_dist)

    def store_transition(self, s, a, r, s_, d):
        self.memory.push(s, a, r, s_, d)

    def soft_update(self, TAU=0.01):
        with torch.no_grad():
            for targetParam, evalParam in zip(self.critic_target.parameters(), self.critic.parameters()):
                targetParam.copy_((1 - self.tau)*targetParam.data + self.tau*evalParam.data)

    def learn(self):
        # sample batch memory from all memory
        b_s, b_a, b_r, b_s_, b_d = self.memory.sample_torch(self.batch_size, self.device) 
        # Critic loss
        with torch.no_grad():
            # V(s) = pi(s)^T [Q(s) - alpha*log(pi(s))]
            a_next = self.actor(b_s_).detach()
            q_next_target = (a_next * (self.critic_target(b_s_) - self.alpha * torch.log(a_next+1e-10))).sum(1, keepdim=True)
            q_target = b_r + (1-b_d) * self.gamma * q_next_target   # TD-target
            
        q_eval = self.critic(b_s).gather(1, b_a)
        #print(q_eval[0], q_target[0])
        self.critic_loss = F.smooth_l1_loss(q_eval, q_target)
        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optim.step()

        # Actor loss
        a_curr = self.actor(b_s)
        q_curr = self.critic(b_s).detach()
        # pi(s)^T [alpha*log(pi(s)) - Q(s)]
        self.actor_loss = (a_curr * (self.alpha*torch.log(a_curr+1e-10) - q_curr)).sum(1).mean()
        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optim.step()

        self.soft_update()
        
        # Adaptive entropy adjustment
        if self.auto_entropy_tuning:
            # J(alpha) = pi(s)^T [-alpha * (log(pi(s)) + H)]
            pi_curr = self.actor(b_s)
            pi = pi_curr.detach()
            alpha = self.log_alpha.exp()
            alpha_loss = pi * (-alpha * (torch.log(pi+1e-10) + self.target_entropy))
            alpha_loss = alpha_loss.sum(1).mean() # dot product            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp().detach().cpu().numpy())

            curr_entropy = -torch.sum(pi * torch.log(pi + 1e-10), 1).mean()
            self.entropy = curr_entropy.cpu().numpy()

        return float(self.actor_loss.detach().cpu().numpy()), float(self.critic_loss.detach().cpu().numpy())
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        anet_path = os.path.join(path, "sac_anet.pt")
        cnet_path = os.path.join(path, "sac_cnet.pt")
        torch.save(self.critic.state_dict(), cnet_path)
        torch.save(self.actor.state_dict(), anet_path)

    def load_model(self, path):
        anet_path = os.path.join(path, "sac_anet.pt")
        cnet_path = os.path.join(path, "sac_cnet.pt")
        self.critic.load_state_dict(torch.load(cnet_path, map_location=self.device))
        self.critic_target.load_state_dict(torch.load(cnet_path, map_location=self.device))
        self.actor.load_state_dict(torch.load(anet_path, map_location=self.device))