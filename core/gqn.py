import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import GQNTower
import generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GQN(nn.Module):
    def __init__(self, ch=64, csize=256, vsize=7, draw_layers=6, down_size=4, share_core=False):
        super(GQN, self).__init__()
        self.vsize = vsize
        self.csize = csize
        self.down_size = down_size
        self.encoder = GQNTower(vsize, csize).to(device)
        self.generator = generator.GeneratorNetwork(x_dim=3, r_dim=csize+vsize, L=draw_layers, scale=down_size, share=share_core).to(device)

    def step_scene_fusion(self, x, v, n_obs):
        feature = self.encoder(x, v)
        scene_rep = torch.sum(feature.view(-1, n_obs, self.csize, feature.shape[2], feature.shape[3]), 1, keepdim=False)
        return scene_rep

    def step_query_view(self, scene_rep, xq, vq):
        pose_code = vq.view(vq.shape[0], -1, 1, 1)
        pose_code = pose_code.repeat(1, 1, scene_rep.shape[2], scene_rep.shape[3])
        scene_rep_pose = torch.cat((scene_rep, pose_code), dim=1)
        x_query, kl = self.generator(xq, scene_rep_pose)
        return x_query, kl

    def step_query_view_sample(self, scene_rep, vq):
        pose_code = vq.view(vq.shape[0], -1, 1, 1)
        pose_code = pose_code.repeat(1, 1, scene_rep.shape[2], scene_rep.shape[3])
        scene_rep_pose = torch.cat((scene_rep, pose_code), dim=1) 
        x_query = self.generator.sample((scene_rep.shape[2]*self.down_size, scene_rep.shape[3]*self.down_size), scene_rep_pose)
        return x_query
    
    def forward(self, x, v, xq, vq, n_obs=3):
        # Scene Fusion
        scene_rep = self.step_scene_fusion(x, v, n_obs)
        # Query Image
        x_query, kl = self.step_query_view(scene_rep, xq, vq)
        return x_query, kl

    def sample(self, x, v, vq, n_obs=3, steps=None):
        # Scene Fusion
        scene_rep = self.step_scene_fusion(x, v, n_obs)
        # Query Image
        x_query = self.step_query_view_sample(scene_rep, vq)
        return x_query