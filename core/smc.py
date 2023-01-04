import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MemoryController(nn.Module):
    def __init__(self, n_wrd_cells, view_size=(16,16), vsize=7, wcode_size=3, emb_size=32, csize=128):
        super(MemoryController, self).__init__()
        self.n_wrd_cells = n_wrd_cells
        self.view_size = view_size
        self.vsize = vsize
        self.emb_size = emb_size
        self.csize = csize
        self.wcode_size = wcode_size

        # Camera Space Embedding / Frustum Activation / Occlusion
        self.fc1 = nn.Linear(wcode_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_act = nn.Linear(256, 1)
        self.fc_emb = nn.Linear(256, emb_size)

        # View Space Embedding Network
        self.vse = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, emb_size)
        )

    def wcode2cam(self, v, wcode):
        with torch.no_grad():
            # Transformation
            vzeros = torch.zeros_like(v[:,0:1])
            vones = torch.ones_like(v[:,0:1])
            v_mtx = torch.cat( [\
                v[:,4:5], -v[:,3:4], vzeros, v[:,0:1], \
                v[:,3:4], v[:,4:5], vzeros, v[:,1:2], \
                vzeros, vzeros, vones , vzeros,
                vzeros, vzeros, vzeros, vones ], 1)

            v_mtx = v_mtx.reshape(-1, 4, 4)
            v_mtx_inv = torch.linalg.inv(v_mtx)
            v_mtx_inv_tile = torch.unsqueeze(v_mtx_inv, 1).repeat(1, self.n_wrd_cells, 1, 1)

            wcode = torch.cat([wcode, torch.ones_like(wcode[:,:1])], 1)
            wcode_batch = torch.unsqueeze(wcode, 0).repeat(v.shape[0], 1, 1)

            wcode_batch_trans = torch.matmul(v_mtx_inv_tile, torch.unsqueeze(wcode_batch, 3))
            wcode_batch_trans = (wcode_batch_trans[...,0])[...,:3]

            '''
            # Visualize
            import numpy as np
            import cv2
            canvas_w = np.ones((512,512,3))
            scale = 32
            # Cam
            #cv2.circle(canvas_w, (256, 256), 5, (0,0,1), 2)
            cam_pos = (int(256+scale*v[0,0]), int(256+scale*v[0,1]))
            cv2.circle(canvas_w, cam_pos, 5, (0,0,0), 2)
            cv2.line(canvas_w, cam_pos, (int(cam_pos[0]+16*v[0,4]), int(cam_pos[1]+16*v[0,3])), (0,0,0), 2)
            # Code
            cv2.circle(canvas_w, (int(256+scale*wcode[0,0]), int(256+scale*wcode[0,1])), 5, (1,0,0), 2)
            cv2.imshow("world", canvas_w)
            ###
            canvas_c = np.ones((512,512,3))
            # Cam
            cv2.circle(canvas_c, (256, 256), 5, (0,0,0), 2)
            cv2.line(canvas_c, (256, 256), (256+int(16*np.cos(0)),256+int(16*np.sin(0))), (0,0,0), 2)
            # Code
            wcode_pos = (wcode_batch_trans[0,0,0], wcode_batch_trans[0,0,1])
            cv2.circle(canvas_c, (int(256+scale*wcode_pos[0]), (int(256+scale*wcode_pos[1]))), 5, (1,0,0), 2)
            cv2.imshow("cam", canvas_c)
            ###
            #print(v[0], wcode[0], wcode_batch[0,0])
            k = cv2.waitKey(0)
            if k == ord('q'):
                exit()
            '''
        return wcode_batch_trans

    def transform(self, wcode_batch_trans, view_size=None):
        if view_size is None:
            view_size = self.view_size

        # Get Transform Location Code of World Cells
        h = F.relu(self.fc1(wcode_batch_trans.reshape(self.n_wrd_cells*wcode_batch_trans.shape[0],-1)))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        activation = torch.sigmoid(self.fc_act(h).view(-1, self.n_wrd_cells, 1))
        cs_embedding = self.fc_emb(h).view(-1, self.n_wrd_cells, self.emb_size)
        
        # View Space Embedding
        x = torch.linspace(-1, 1, view_size[0])
        y = torch.linspace(-1, 1, view_size[1])
        x_grid, y_grid = torch.meshgrid(x, y)
        vcode = torch.cat((torch.unsqueeze(x_grid, 0), torch.unsqueeze(y_grid, 0)), dim=0).reshape(2,-1).permute(1,0).to(device) #(16*16, 2)
        vs_embedding = self.vse(vcode) #(256, 128)
        vs_embedding = torch.unsqueeze(vs_embedding, 0).repeat(wcode_batch_trans.shape[0], 1, 1) #(-1, view_cell, emb_size)
        
        # Cross-Space Cell Relation
        relation = torch.bmm(cs_embedding, vs_embedding.permute(0,2,1)) #(-1, wrd_cell, view_cell)
        return relation, activation

    def get_clip_tensor(self, wcode_batch, clip_range):
        with torch.no_grad():
            clip1 = (wcode_batch[:,:,0]<clip_range).float()
            clip2 = (wcode_batch[:,:,0]>-clip_range).float()
            clip3 = (wcode_batch[:,:,1]<clip_range).float()
            clip4 = (wcode_batch[:,:,1]>-clip_range).float()
            w_clip = clip1 * clip2 * clip3 * clip4
        return w_clip

    def forward(self, view_cell, v, wcode, view_size=None, clip_range=3):
        if view_size is None:
            view_size = self.view_size
        wcode_batch_trans = self.wcode2cam(v, wcode)
        relation, activation = self.transform(wcode_batch_trans, view_size=view_size)

        distribution = torch.softmax(relation, 2)
        route = distribution * activation   # (-1, n_wrd_cells, n_view_cells)
        wrd_cell = torch.bmm(view_cell, route.permute(0,2,1))

        # Clip the out-of-range cells.
        if clip_range is not None:
            w_clip = self.get_clip_tensor(wcode_batch_trans, clip_range)
            w_mask = w_clip.float().unsqueeze(1)
            wrd_cell = wrd_cell * w_mask

        return wrd_cell # (-1, csize, n_wrd_cells)
    
    def query(self, wrd_cell, v, wcode, view_size=None):
        if view_size is None:
            view_size = self.view_size

        wcode_batch_trans = self.wcode2cam(v, wcode)
        relation, activation = self.transform(wcode_batch_trans, view_size=view_size)

        distribution = torch.softmax(relation, 1)
        route = distribution * activation   # (-1, n_wrd_cells, n_view_cells)
        query_view_cell = torch.bmm(wrd_cell, route).reshape(-1, self.csize, view_size[0], view_size[1])
        return query_view_cell