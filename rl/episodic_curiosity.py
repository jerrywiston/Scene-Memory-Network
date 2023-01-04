import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import replay_memory

class EmbeddingNet(nn.Module):
    def __init__(self, input_shape, emb_dim=32):
        super(EmbeddingNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, emb_dim),
            nn.ReLU(),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, s):
        obs = s["obs"]
        conv_out = self.conv(obs)
        emb = self.fc(conv_out)
        return emb

class ActionPredictor(nn.Module):
    def __init__(self, emb_dim, n_actions):
        super(ActionPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim*2, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    
    def forward(self, emb1, emb2):
        emb_cat = torch.cat([emb1, emb2], 1)
        logits = self.fc(emb_cat)
        prob = F.softmax(logits)
        return prob, logits

class EpisodicCuriosityEuclidean:
    def __init__(
        self,
        n_actions,
        input_shape,
        device,
        batch_size = 32,
        emb_dim = 32,
        memory_size = 5000
    ):
    self.device = device
    self.embnet = EmbeddingNet(input_shape, emb_dim)
    self.classifier = ActionPredictor(emb_dim, n_actions)
    self.parameters = self.embnet.parameters() + self.classifier.parameters()
    self.optimizer = optim.Adam(self.parameters, lr=1e-4)
    self.memory = replay_memory.Memory(memory_size)
    self.criterion = nn.CrossEntropyLoss()
    self.dqueue = []
    self.dqueue_size = 64
    self.episodic_memory = []

    def store_transition(self, s, a, r, s_):
        self.memory.push(s, a, r, s_, d)
    
    def store_episodic_memory(self, s):
        b_s = torch.FloatTensor(s[np.newaxis,...]).to(self.device)
        e = self.embnet(b_s).detach().cpu().numpy()
        self.episodic_memory.append(e)

    def train_embedding(self):
        b_s, b_a, b_r, b_s_, b_d = self.memory.sample_torch(self.batch_size, self.device) 
        e1 = self.embnet(b_s)
        e2 = self.embnet(b_s_)
        prob, logits = self.classifier(e1, e2)
        #
        self.optimizer.zero_grad()
        loss = self.criterion(logits, b_a)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()    

    def kernal(self, x, y, eps=1e-3):
        d = np.linalg.norm(x - y)
        dqueue.append(d2)
        if len(dqueue) > dqueue_size:
            dqueue.pop(0)
        dm = np.array(dmqueue).mean()
        k = eps / (d**2/dm**2 + eps) 
        return k

    def get_nearest_id(self, s):

    def get_intrinsic_reward(self, s, n=32):
        pass