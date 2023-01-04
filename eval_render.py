import numpy as np
import cv2
import torch

import maze3d.maze_env as menv
from maze3d import maze
from maze3d.gen_maze_dataset import gen_data_global, gen_data_range

from core.smn import SMN
from core.gqn import GQN
from core.strgqn import STRGQN
from maze_rep_env import RenderModel

import configparser
import config_handle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############ Utility Functions ############
def get_batch(color, pose, obs_size=12, batch_size=32, to_torch=True):
    img_obs = None
    pose_obs = None
    img_query = None
    pose_query = None
    for i in range(batch_size):
        batch_id = np.random.randint(0, color.shape[0])
        obs_id = np.random.randint(0, color.shape[1], size=obs_size)
        query_id = np.random.randint(0, color.shape[1])
        
        if img_obs is None:
            img_obs = color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])
            pose_obs = pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])
            img_query = color[batch_id:batch_id+1, query_id]
            pose_query = pose[batch_id:batch_id+1, query_id]
        else:
            img_obs = np.concatenate([img_obs, color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])], 0)
            pose_obs = np.concatenate([pose_obs, pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])], 0)
            img_query = np.concatenate([img_query, color[batch_id:batch_id+1, query_id]], 0)
            pose_query = np.concatenate([pose_query, pose[batch_id:batch_id+1, query_id]], 0)
    
    if to_torch:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_obs = torch.FloatTensor(img_obs).permute(0,3,1,2).to(device)
        pose_obs = torch.FloatTensor(pose_obs).to(device)
        img_query = torch.FloatTensor(img_query).permute(0,3,1,2).to(device)
        pose_query = torch.FloatTensor(pose_query).to(device)

    return img_obs, pose_obs, img_query, pose_query

def draw_query_maze(net, color, pose, obs_size=3, row_size=32, gen_size=10, shuffle=False, border=[1,4,3]):
    img_list = []
    for it in range(gen_size):
        img_row = []
        x_obs, v_obs, x_query_gt, v_query = get_batch(color, pose, obs_size, row_size)
        img_size = (x_obs.shape[-2], x_obs.shape[-1])
        vsize = v_obs.shape[-1]
        with torch.no_grad():
            x_query_sample = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
            x_query_sample = x_query_sample.detach().permute(0,2,3,1).cpu().numpy()
        
        for j in range(x_query_sample.shape[0]):
            x_np = []
            bscale = int(img_size[1]/64)
            for i in range(obs_size):
                x_np.append(x_obs[obs_size*j+i].detach().permute(1,2,0).cpu().numpy())
                if i < obs_size-1:
                    x_np.append(np.ones([img_size[0],border[0]*bscale,3]))
            x_np.append(np.ones([img_size[0],border[1]*bscale,3]))
            x_np.append(x_query_gt.detach().permute(0,2,3,1).cpu().numpy()[j])
            x_np.append(np.ones([img_size[0],border[1]*bscale,3]))
            x_np.append(x_query_sample[j])
            x_np = np.concatenate(x_np, 1)
            img_row.append(x_np)

            if j < row_size:
                img_row.append(np.ones([border[2]*bscale,x_np.shape[1],3]))
        
        img_row = np.concatenate(img_row, 0) * 255
        #img_row = cv2.cvtColor(img_row.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_list.append(img_row.astype(np.uint8))
        fill_size = len(str(gen_size))
        print("\rProgress: "+str(it+1).zfill(fill_size)+"/"+str(gen_size), end="")
    print()
    return img_list

class RenderModelGQN:
    def build_model(self, args):
        self.cell_dim = args.c
        args.img_size = (args.v[0]*args.down_size, args.v[1]*args.down_size)
        self.net = GQN(csize=args.c, ch=args.ch, vsize=args.vsize, draw_layers=args.draw_layers, \
                    down_size=args.down_size, share_core=args.share_core).to(device)
        self.scene_rep_size = [args.c, args.v[0], args.v[1]]

    def init_scene_rep(self):
        self.scene_code = torch.zeros(self.scene_rep_size, device=device)

    def load_parameters(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
    
    def add_observation(self, x, v):
        with torch.no_grad():
            self.scene_code = self.scene_code + self.net.encoder(x, v)

    def query_view(self, v):
        with torch.no_grad():
            x_render = self.net.step_query_view_sample(self.scene_code, v)
            return x_render

class RenderModelGTMSM:
    def build_model(self, args):
        self.cell_dim = args.c
        args.img_size = (args.v[0]*args.down_size, args.v[1]*args.down_size)
        self.net = GQN(csize=args.c, ch=args.ch, vsize=args.vsize, draw_layers=args.draw_layers, \
                    down_size=args.down_size, share_core=args.share_core).to(device)
        self.eps_memory = []
        self.scene_rep_size = [args.c, args.v[0], args.v[1]]

    def init_scene_rep(self):
        self.eps_memory = []
        self.scene_code = torch.zeros(self.scene_rep_size, device=device)

    def load_parameters(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
    
    def add_observation(self, x, v):
        self.eps_memory.append({"x":x, "v":v})
        
    def query_view(self, v, knn=5):
        with torch.no_grad():
            dist_rec = []
            for data in self.eps_memory:
                v_obs = data["v"]
                dist = (v_obs[0,0]-v[0,0])**2 + (v_obs[0,1]-v[0,1])**2 + (v_obs[0,2]-v[0,2])**2
                dist_rec.append(dist)
            eps_memory_sort = [x for _,x in sorted(zip(dist_rec,self.eps_memory))]
            self.scene_code = torch.zeros(self.scene_rep_size, device=device)
            for i in range(knn):
                self.scene_code = self.scene_code + self.net.encoder(eps_memory_sort[i]["x"], eps_memory_sort[i]["v"])
            x_render = self.net.step_query_view_sample(self.scene_code, v)
            return x_render

class RenderModelSTRGQN:
    def build_model(self, args):
        self.cell_dim = args.c
        args.img_size = (args.v[0]*args.down_size, args.v[1]*args.down_size)
        self.net = STRGQN(n_wrd_cells=args.w, view_size=args.v, csize=args.c, ch=args.ch, vsize=args.vsize, \
            draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)
    
    def init_scene_rep(self, samp_size=2048, cell_dim=32):
        self.scene_cell = torch.zeros((1, cell_dim, samp_size), device=device)

    def load_parameters(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
    
    def add_observation(self, x, v):
        with torch.no_grad():
            self.scene_cell = self.scene_cell + self.net.step_observation_encode(x, v)

    def query_view(self, v):
        with torch.no_grad():
            scene_cell_prob = torch.sigmoid(self.scene_cell)
            x_render = self.net.step_query_view_sample(scene_cell_prob, v)
            return x_render

if __name__ == "__main__":
    # Select maze type.
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='?', type=str, default="maze" ,help='Experiment name.')
    parser.add_argument('--render_core', nargs='?', type=str, default="smn", help='smn/strgqn/gqn/gtmsm')
    parser.add_argument('--eval_type', nargs='?', type=str, default="local", help='local/base/global')
    path = parser.parse_args().path
    render_core = parser.parse_args().render_core
    eval_type = parser.parse_args().eval_type
    if render_core == "strgqn":
        render_model = RenderModelSTRGQN()
    elif render_core == "gqn":
        render_model = RenderModelGQN()
    elif render_core == "smn":
        render_model = RenderModel()
    elif render_core == "gtmsm":
        render_model = RenderModelGTMSM()
    else:
        print("Invalid rendering core!")
        exit(0)
    config = configparser.ConfigParser()
    config.read(os.path.join(path, "config.conf"))
    args = config_handle.get_config_strgqn(config)
    render_model.build_model(args)
    save_path = os.path.join(path, "save")
    render_model.load_parameters(os.path.join(save_path, "model.pth"))
    ##################
    if eval_type == "large":
        size = (17,17)
    else:
        size = (11,11)
    maze_obj = maze.MazeGridRandom2(size=size, room_total=5) # (11, 11)(17, 17)
    env = menv.MazeBaseEnv(maze_obj, render_res=(64,64))
    # Local(5,10) / Base(10,10) / (15,10)
    if eval_type == "local":
        obs_size = 5
    elif eval_type == "base":
        obs_size = 10
    elif eval_type == "large":
        obs_size = 20
    else:
        print("Invalid evaluation type.")
        exit(0)
    query_size = 10
    ##################
    l1_loss_rec = []
    l2_loss_rec = []
    for i in range(100):
        print(i)
        if render_core == "smn":
            render_model.init_scene_cell([-5,20,-5,20,-3,3], 16, 32)
        else:
            render_model.init_scene_rep()
        path = None#"level_save/" + str(i).zfill(4) + ".npz"
        if eval_type == "local":
            color, depth, pose = gen_data_range(env, 3, obs_size+query_size, path)
        else:
            color, depth, pose = gen_data_global(env, obs_size+query_size, path)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for j in range(obs_size):
            img_obs = torch.FloatTensor(color[j:j+1]/255.).permute(0,3,1,2).to(device)
            pose_obs = torch.FloatTensor(pose[j:j+1]).to(device)
            render_model.add_observation(img_obs, pose_obs)

        for k in range(obs_size, obs_size+query_size):
            img_query = color[k].astype(np.float)
            pose_query = torch.FloatTensor(pose[k:k+1]).to(device)
            img_query_render = render_model.query_view(pose_query)[0].permute(1,2,0).detach().cpu().numpy()
            img_query_render = (img_query_render*255).astype(np.float)
            l1_loss = np.abs(img_query - img_query_render).mean()   
            l2_loss = np.sqrt(np.square(img_query - img_query_render).mean())

            l1_loss_rec.append(l1_loss)
            l2_loss_rec.append(l2_loss)
            '''
            img_query = cv2.resize(img_query, (256,256), interpolation=cv2.INTER_NEAREST)
            img_query_render = cv2.resize(img_query_render, (256,256), interpolation=cv2.INTER_NEAREST)
            img_draw = np.concatenate([img_query, img_query_render], 1)
            cv2.imshow("test", img_draw)
            k = cv2.waitKey(0)
            if k==ord('q'):
                exit(0)
            '''

    l1_loss_rec = np.array(l1_loss_rec)
    l2_loss_rec = np.array(l2_loss_rec)
    #print(l1_loss_rec)
    print(l1_loss_rec.mean(), l1_loss_rec.std()) 

    #print(l2_loss_rec)
    print(l2_loss_rec.mean(), l2_loss_rec.std())


        

