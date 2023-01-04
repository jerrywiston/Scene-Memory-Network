import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tsm import SpatialBlock, TSMBlock

EMB_DIM = 64
VAL_DIM = 64
KEY_DIM = 32

class FRMQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(FRMQN, self).__init__()
        self.conv = nn.Sequential(
            SpatialBlock(frame_dim=3, out_dim=32, kernal_size=7, stride=2),
            nn.ReLU(),
            SpatialBlock(frame_dim=32, out_dim=32, kernal_size=5, stride=2),
            nn.ReLU(),
            SpatialBlock(frame_dim=32, out_dim=64, kernal_size=3, stride=2),
            nn.ReLU(),
            SpatialBlock(frame_dim=64, out_dim=64, kernal_size=3, stride=2),
            nn.ReLU(),
        )
        feat_dim = self._get_conv_feature_size(input_shape)
        self.conv_embedding = nn.Conv1d(feat_dim, EMB_DIM, 1)
        
        # Memory Related
        self.Wkey = nn.Conv1d(EMB_DIM, KEY_DIM, 1)
        self.Wval = nn.Conv1d(EMB_DIM, VAL_DIM, 1)
        
        # Context Related
        self.context_rnn = torch.nn.GRUCell(EMB_DIM, KEY_DIM)

        # Duel Network
        self.fc_advantage = nn.Linear(EMB_DIM+KEY_DIM, n_actions)
        self.fc_value = nn.Linear(EMB_DIM+KEY_DIM, 1)

    def _get_conv_feature_size(self, shape):
        o = self.conv(torch.zeros(1, *shape)).reshape([1, int(shape[0]/3), -1])
        return int(o.size()[2])
    
    def forward(self, s):
        obs = s["obs"]
        frames = int(obs.shape[1] / 3)

        # Extract observatoin embedding
        conv_out = self.conv(obs).reshape([obs.shape[0], frames, -1])
        obs_emb = torch.relu(self.conv_embedding(conv_out.permute(0,2,1))) # (batch, emb_dim, frames)
        
        # Constuct memory
        m_val = self.Wval(obs_emb) # (batch, val_dim, frames)
        m_key = self.Wkey(obs_emb) # (batch, key_dim, frames)

        # Context rnn
        h = torch.zeros(obs.shape[0],KEY_DIM).to(obs.get_device())
        for i in range(frames):
            h = self.context_rnn(obs_emb[:,:,i], h) # (batch, key_dim)
        h_query = torch.unsqueeze(h, -1)
        att = torch.nn.Softmax(dim=1)(torch.bmm(m_key.permute(0,2,1), h_query)) # (batch, frames, 1)
        o = torch.matmul(m_val, att).reshape(obs.shape[0],-1)

        # Compute Q
        advantage = self.fc_advantage(torch.cat([h,o],1))
        value = self.fc_value(torch.cat([h,o],1))
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q
