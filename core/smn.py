import torch
import torch.nn as nn
import torch.nn.functional as F
import encoder
#import strn_crop as strn
import smc
import generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Default Setting
# ==============================
# Image Size: (3, 64, 64)
# Cell Channels: 128
# View Cells: (16, 16)
# World Cells: 2000
# Draw Layers: 6
# ==============================

# Deterministic
class SMN(nn.Module):
    def __init__(self, n_wrd_cells=2000, view_size=(16,16), csize=128, ch=64, vsize=7, draw_layers=6, down_size=4, wcode_size=3, share_core=False):
        super(SMN, self).__init__()
        self.n_wrd_cells = n_wrd_cells
        self.view_size = view_size
        self.csize = csize
        self.vsize = vsize
        self.ch = ch
        self.down_size = down_size
        self.draw_layers = draw_layers
        self.wcode_size = wcode_size # Must be 3 for "trans" version.

        self.encoder = encoder.EncoderNetworkRes(ch, csize, down_size).to(device)
        self.strn = smc.MemoryController(n_wrd_cells, view_size=view_size, vsize=vsize, csize=csize).to(device)
        self.generator = generator.GeneratorNetwork(x_dim=3, r_dim=csize, L=draw_layers, scale=down_size, share=share_core).to(device)
        self.sample_wcode(samp_size=self.n_wrd_cells)

    def sample_wcode(self, samp_size=None, scale=[3.0, 3.0, 3.0]):
        if samp_size == None:
            samp_size = self.n_wrd_cells
        self.wcode = torch.rand(samp_size, self.wcode_size).to(device)
        with torch.no_grad():
            self.wcode[:,0] = (self.wcode[:,0] * 2 - 1) * scale[0]
            self.wcode[:,1] = (self.wcode[:,1] * 2 - 1) * scale[1]
            self.wcode[:,2] = (self.wcode[:,2] * 2 - 1) * scale[2]

    def step_observation_encode(self, x, v, view_size=None):
        if view_size is None:
            view_size = self.view_size
        view_cell = self.encoder(x).reshape(-1, self.csize, view_size[0]*view_size[1])
        wrd_cell = self.strn(view_cell, v, self.wcode, view_size=view_size)
        return wrd_cell
    
    def step_scene_fusion(self, wrd_cell, n_obs): 
        scene_cell = torch.sum(wrd_cell.view(-1, n_obs, self.csize, self.n_wrd_cells), 1, keepdim=False)
        #scene_cell = torch.sigmoid(scene_cell)
        return scene_cell
    
    def step_query_view(self, scene_cell, xq, vq):
        view_cell_query = self.strn.query(scene_cell, vq, self.wcode)
        view_cell_query = torch.sigmoid(view_cell_query) ###
        x_query, kl = self.generator(xq, view_cell_query)
        return x_query, kl

    def step_query_view_sample(self, scene_cell, vq):
        view_cell_query = self.strn.query(scene_cell, vq, self.wcode)
        view_cell_query = torch.sigmoid(view_cell_query) ###
        sample_size = (self.view_size[0]*self.down_size, self.view_size[1]*self.down_size)
        x_query = self.generator.sample(sample_size, view_cell_query)
        return x_query
    
    def visualize_routing(self, view_cell, v, vq, view_size=None):
        if view_size is None:
            view_size = self.view_size
        wrd_cell = self.strn(view_cell.reshape(-1, self.csize, view_size[0]*view_size[1]), v, self.wcode, view_size=view_size)
        scene_cell = wrd_cell
        view_cell_query = self.strn.query(scene_cell, vq, view_size=view_size)
        return view_cell_query

    def forward(self, x, v, xq, vq, n_obs=3):
        # Move to the coordinate of query view
        with torch.no_grad():
            vq_tile = vq.unsqueeze(1).repeat(1,n_obs,1).reshape(v.shape[0],-1)
            v[:,0:3] = v[:,0:3] - vq_tile[:,0:3]
            vq[:,0:3] = 0

        # Observation Encode
        wrd_cell = self.step_observation_encode(x, v)
        # Scene Fusion
        scene_cell = self.step_scene_fusion(wrd_cell, n_obs)
        # Query Image
        x_query, kl = self.step_query_view(scene_cell, xq, vq)
        return x_query, kl

    def sample(self, x, v, vq, n_obs=3, steps=None):
        # Move to the coordinate of query view
        with torch.no_grad():
            vq_tile = vq.unsqueeze(1).repeat(1,n_obs,1).reshape(v.shape[0],-1)
            v[:,0:3] = v[:,0:3] - vq_tile[:,0:3]
            vq[:,0:3] = 0
            
        # Observation Encode
        wrd_cell = self.step_observation_encode(x, v)
        # Scene Fusion
        scene_cell = self.step_scene_fusion(wrd_cell, n_obs)
        # Query Image
        x_query = self.step_query_view_sample(scene_cell, vq)
        return x_query

    def reconstruct(self, x):
        view_cell = self.encoder(x)
        view_cell = torch.sigmoid(view_cell)
        random_mask = torch.rand((x.size(0),1,self.view_size[0],self.view_size[1])).to(device)
        view_cell_mask = random_mask * view_cell
        x_rec, kl = self.generator(x, view_cell_mask)
        return x_rec, kl

    #############################
    # Demo
    #############################
    def construct_scene_representation(self, x, v):
        self.wrd_cell_record = self.step_observation_encode(x, v)
        self.scene_cell_record = torch.sigmoid(torch.sum(self.wrd_cell_record, 0, keepdim=True))

    def scene_render(self, vq, obs_act=None, noise=False):
        self.scene_cell_record = torch.zeros_like(self.scene_cell_record)
        if obs_act is not None:
            for i in range(obs_act.shape[0]):
                if obs_act[i] == 0:
                    continue
                self.scene_cell_record += self.wrd_cell_record[i:i+1]
        self.scene_cell_record = torch.sigmoid(self.scene_cell_record)
        sample_size = (self.view_size[0]*self.down_size, self.view_size[1]*self.down_size)
        view_cell_query = self.strn.query(self.scene_cell_record, vq)
        sample_size = (self.view_size[0]*self.down_size, self.view_size[1]*self.down_size)
        x_query = self.generator.sample(sample_size, view_cell_query, noise)
        #
        return x_query