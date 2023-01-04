import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STRN(nn.Module):
    def __init__(self, n_wrd_cells, view_size=(16,16), vsize=7, wcode_size=3, emb_size=32, csize=128):
        super(STRN, self).__init__()
        self.n_wrd_cells = n_wrd_cells
        self.view_size = view_size
        self.wcode_size = wcode_size
        self.emb_size = emb_size
        self.csize = csize

        # Location Code Extracting for World Cells
        self.w2c = nn.Sequential(
            nn.Linear(vsize, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, wcode_size*n_wrd_cells),
        )
        
        # Camera Space Embedding / Frustum Activation / Occlusion
        self.fc1 = nn.Linear(wcode_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_act = nn.Linear(256, 1)
        self.fc_emb = nn.Linear(256, emb_size)

        # View Space Embedding Network
        self.vse = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, emb_size)
        )

    def transform(self, v, view_size=None):
        if view_size is None:
            view_size = self.view_size
        # Camera Space Embedding
        wcode = self.w2c(v).view(-1, self.wcode_size)
        h = F.relu(self.fc1(wcode))
        h = F.relu(self.fc2(h))
        activation = torch.sigmoid(self.fc_act(h).view(-1, self.n_wrd_cells, 1))
        cs_embedding = self.fc_emb(h).view(-1, self.n_wrd_cells, self.emb_size)
        
        # View Space Embedding
        x = torch.linspace(-1, 1, view_size[0])
        y = torch.linspace(-1, 1, view_size[1])
        x_grid, y_grid = torch.meshgrid(x, y)
        vcode = torch.cat((torch.unsqueeze(x_grid, 0), torch.unsqueeze(y_grid, 0)), dim=0).reshape(2,-1).permute(1,0).to(device) #(16*16, 2)
        vs_embedding = self.vse(vcode) #(256, 128)
        vs_embedding = torch.unsqueeze(vs_embedding, 0).repeat(v.shape[0], 1, 1) #(-1, view_cell, emb_size)
        
        # Cross-Space Cell Relation
        relation = torch.bmm(cs_embedding, vs_embedding.permute(0,2,1)) #(-1, wrd_cell, view_cell)
        return relation, activation, wcode

    def forward(self, view_cell, v, view_size=None):
        if view_size is None:
            view_size = self.view_size
        relation, activation, wcode = self.transform(v, view_size=view_size)
        distribution = torch.softmax(relation, 2)
        route = distribution * activation   # (-1, n_wrd_cells, n_view_cells)
        wrd_cell = torch.bmm(view_cell, route.permute(0,2,1))
        return wrd_cell # (-1, csize, n_wrd_cells)
    
    def query(self, wrd_cell, v, view_size=None):
        if view_size is None:
            view_size = self.view_size
        relation, activation, wcode = self.transform(v, view_size=view_size)
        distribution = torch.softmax(relation, 1)
        route = distribution * activation   # (-1, n_wrd_cells, n_view_cells)
        query_view_cell = torch.bmm(wrd_cell, route).reshape(-1, self.csize, view_size[0], view_size[1])
        return query_view_cell