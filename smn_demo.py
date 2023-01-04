import numpy as np
import cv2
import torch

import maze3d.maze_env as menv
from maze3d import maze
from core.smn import SMN

import configparser
import config_handle

GET_CELL = True#False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RenderModel:
    def __init__(self):
        pass

    def init_scene_cell(self, cell_range, cell_density, cell_dim):
        volume = np.abs(cell_range[0] - cell_range[1]) * np.abs(cell_range[2] - cell_range[3]) * np.abs(cell_range[4] - cell_range[5])
        samp_size = cell_density * volume
        self.scene_code = torch.rand(1, samp_size, 3).to(device)
        with torch.no_grad():
            self.scene_code[:,:,0] = self.scene_code[:,:,0] * (cell_range[1] - cell_range[0]) + cell_range[0]
            self.scene_code[:,:,1] = self.scene_code[:,:,1] * (cell_range[3] - cell_range[2]) + cell_range[2]
            self.scene_code[:,:,2] = self.scene_code[:,:,2] * (cell_range[5] - cell_range[4]) + cell_range[4]
        self.scene_cell = torch.zeros((1, samp_size, cell_dim), device=device)
        self.cell_range = cell_range
        
    def build_model(self, args):
        args.img_size = (args.v[0]*args.down_size, args.v[1]*args.down_size)
        self.net = SMN(n_wrd_cells=args.w, view_size=args.v, csize=args.c, ch=args.ch, vsize=args.vsize, \
            draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)
    
    def load_parameters(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

    def get_clip_tensor(self, wcode_batch, clip_range=3):
        with torch.no_grad():
            clip1 = wcode_batch[:,:,0] < clip_range
            clip2 = wcode_batch[:,:,0] > -clip_range
            clip3 = wcode_batch[:,:,1] < clip_range
            clip4 = wcode_batch[:,:,1] > -clip_range
            w_clip = clip1 * clip2 * clip3 * clip4
        return w_clip

    def compute_entropy(self, v):
        with torch.no_grad():
            scene_code_trans = self.scene_code - v[:,0:3]
            crop_mask = self.get_clip_tensor(scene_code_trans)
            prob = torch.sigmoid(self.scene_cell[crop_mask])
            entropy = -prob*torch.log(prob+1e-10) - (1-prob)*torch.log(1-prob+1e-10)
            return entropy.mean().cpu().numpy()

    def get_cell_feature(self, v, clip_range=4, n_cells=1000):
        with torch.no_grad():
            scene_code_trans = self.scene_code - v[:,0:3]
            crop_mask = self.get_clip_tensor(scene_code_trans)
            scene_code_clip = torch.cat([scene_code_trans, v[:,3:].unsqueeze(1).repeat(1,self.scene_code.shape[1],1)], 2)
            scene_code_clip = scene_code_clip[crop_mask]
            prob = torch.sigmoid(self.scene_cell[crop_mask])
            cell_feature = torch.cat([prob, scene_code_clip], 1)
        #return cell_feature.cpu().numpy()[:1000]
        return prob.cpu().numpy()[:n_cells], scene_code_clip.cpu().numpy()[:n_cells]

    def add_observation(self, x, v):
        with torch.no_grad():
            scene_code_trans = self.scene_code - v[:,0:3]
            crop_mask = self.get_clip_tensor(scene_code_trans)
            self.net.wcode = self.scene_code[crop_mask]
            self.net.n_wrd_cells = self.net.wcode.shape[0]
            self.net.strn.n_wrd_cells = self.net.wcode.shape[0]
            self.scene_cell[crop_mask] = self.scene_cell[crop_mask] + self.net.step_observation_encode(x, v).squeeze().permute(1,0)

    def query_view(self, v):
        with torch.no_grad():
            scene_code_trans = self.scene_code - v[:,0:3]
            crop_mask = self.get_clip_tensor(scene_code_trans, 3)
            #print(crop_mask.sum())
            self.net.wcode = self.scene_code[crop_mask]
            self.net.n_wrd_cells = self.net.wcode.shape[0]
            self.net.strn.n_wrd_cells = self.net.wcode.shape[0]
            #scene_cell_prob = torch.sigmoid(self.scene_cell[crop_mask]).permute(1,0).unsqueeze(0)
            scene_cell_prob = self.scene_cell[crop_mask].permute(1,0).unsqueeze(0)
            x_render = self.net.step_query_view_sample(scene_cell_prob, v)
            return x_render

class MazeSceneRepEnv(menv.MazeBaseEnv):
#class MazeSceneRepEnv(menv.MazeNavEnv):
#class MazeSceneRepEnv(menv.MazeItemEnv):
    def __init__(self, 
            maze_obj, 
            exp_path, 
            err_th=600, 
            render_res=(192,192), 
            velocity=0.2,#0.1, 
            ang_velocity=np.pi*2/18,
            cell_range=[-3,14,-3,14,-3,3], #[-3,54,-3,54,-3,3], #[-5,20,-5,20,-3,3], 
            cell_density=16, 
            fov=80*np.pi/180,
            discrete_control=True,
            n_items = 15
        ):
        #super().__init__(maze_obj, render_res, velocity, ang_velocity, fov=80*np.pi/180, discrete_control=discrete_control, n_items=n_items)
        super().__init__(maze_obj, render_res, velocity, ang_velocity, fov=80*np.pi/180, discrete_control=discrete_control)
        self.args = self._load_config(exp_path)
        self.cell_range = cell_range
        self.cell_density = cell_density
        self._init_render_model(self.args, self.cell_range, self.cell_density)
        self.err_th = err_th
        self.obs_list = []

    def _load_config(self, exp_path):
        self.exp_path = exp_path
        config_file = self.exp_path + "config.conf"
        config = configparser.ConfigParser()
        config.read(config_file)
        args = config_handle.get_config_strgqn(config)
        return args
    
    def _init_render_model(self, args, cell_range, cell_density):
        save_path = self.exp_path + "save/"
        self.render_model = RenderModel()
        self.render_model.init_scene_cell(cell_range, cell_density, cell_dim=args.c)
        self.render_model.build_model(args)
        self.render_model.load_parameters(save_path+"model.pth")

    def _add_obs(self, state, info, keyframe=False):
        x = cv2.resize(state, (64,64)).astype(float) / 255.
        v = np.array([info["pose"][0], info["pose"][1], 0, np.sin(info["pose"][2]), np.cos(info["pose"][2]), 0, 0])
        x = torch.FloatTensor(x).unsqueeze(0).permute(0,3,1,2).to(device)
        v = torch.FloatTensor(v).unsqueeze(0).to(device)
        ent1 = self.render_model.compute_entropy(v)
        self.render_model.add_observation(x,v)
        ent2 = self.render_model.compute_entropy(v)
        if keyframe:
            self.obs_list.append(info["pose"])
        return ent1 - ent2

    def reset(self, gen_maze=True, init_pos_info=None):
        state, info = super().reset(gen_maze, init_pos_info)
        self.traj[len(self.traj)-1]["value"] = 0
        if gen_maze:
            self._init_render_model(self.args, self.cell_range, self.cell_density)
            self.obs_list = []
        # Add Observation
        self._add_obs(state, info, True)
        # Render
        v = np.array([info["pose"][0], info["pose"][1], 0, np.sin(info["pose"][2]), np.cos(info["pose"][2]), 0, 0])
        v = torch.FloatTensor(v).unsqueeze(0).to(device)
        self.x_render = self.render_model.query_view(v).detach().permute(0,2,3,1).squeeze().cpu().numpy()*255
        info["render"] = self.x_render
        
        # Multiscale Cell
        if GET_CELL:
            #info["cell"], info["code"] = self.render_model.get_cell_feature(v)
            #info["cell_s1"], info["code_s1"] = self.render_model.get_cell_feature(v, 2)
            info["cell_s2"], info["code_s2"] = self.render_model.get_cell_feature(v, 4)
            #info["cell_s3"], info["code_s3"] = self.render_model.get_cell_feature(v, 6)
        return state, info
    
    def reset_from_file(self, path):
        state, info = super().reset_from_file(path)
        self.traj[len(self.traj)-1]["value"] = 0
        self._init_render_model(self.args, self.cell_range, self.cell_density)
        self.obs_list = []
        # Add Observation
        self._add_obs(state, info)
        # Render
        v = np.array([info["pose"][0], info["pose"][1], 0, np.sin(info["pose"][2]), np.cos(info["pose"][2]), 0, 0])
        v = torch.FloatTensor(v).unsqueeze(0).to(device)
        self.x_render = self.render_model.query_view(v).detach().permute(0,2,3,1).squeeze().cpu().numpy()*255
        info["render"] = self.x_render
        
        # Multiscale Cell
        if GET_CELL:
            #info["cell"], info["code"] = self.render_model.get_cell_feature(v)
            #info["cell_s1"], info["code_s1"] = self.render_model.get_cell_feature(v, 2)
            info["cell_s2"], info["code_s2"] = self.render_model.get_cell_feature(v, 4)
            #info["cell_s3"], info["code_s3"] = self.render_model.get_cell_feature(v, 6)
        return state, info

    
    def step(self, action, add_obs=True):
        state_next, reward, done, info = super().step(action)
        if add_obs:
            ent_diff = self._add_obs(state_next, info, True)
        else:
            ent_diff = 0
        # Render View
        v = np.array([info["pose"][0], info["pose"][1], 0, np.sin(info["pose"][2]), np.cos(info["pose"][2]), 0, 0])
        v = torch.FloatTensor(v).unsqueeze(0).to(device)
        self.x_render = self.render_model.query_view(v).detach().permute(0,2,3,1).squeeze().cpu().numpy()*255
        x_gt_re = cv2.resize(state_next, (64,64), interpolation=cv2.INTER_NEAREST)
        err = ((self.x_render - x_gt_re)**2).mean()
        # Add Observation
        kf = False
        ent_diff = 0
        '''
        if err > self.err_th:
            # Check History
            is_redundant = False
            for obs in self.obs_list:
                dist = np.sqrt((info["pose"][0] - obs[0])**2 + (info["pose"][1] - obs[1])**2)
                ang_dist = np.abs(np.rad2deg(info["pose"][2] - obs[2]) % 360)
                if dist < 0.3 and ang_dist < 20:
                    is_redundant = True
                    break
            if not is_redundant:
                reward = 1.0
                klf = True
                reward = 10*ent_diff
        '''
        
        int_reward = ent_diff*10
        draw_value = 100*ent_diff
        if draw_value > 1:
            self.traj[len(self.traj)-1]["value"] = 1
        else:
            self.traj[len(self.traj)-1]["value"] = draw_value
        info["render"] = self.x_render
        
        # Navigation Reward
        #if "goal" in info:
        #    if info["goal"]:
        #        reward = 15 * reward#10
        
        #reward = 15*reward + int_reward
        reward = 5*reward + int_reward
                
        # Multiscale Cell
        if GET_CELL:
            #info["cell"], info["code"] = self.render_model.get_cell_feature(v)
            #info["cell_s1"], info["code_s1"] = self.render_model.get_cell_feature(v, 2)
            info["cell_s2"], info["code_s2"] = self.render_model.get_cell_feature(v, 4)
            #info["cell_s3"], info["code_s3"] = self.render_model.get_cell_feature(v, 6)
        return state_next, reward, done, info
    
    def render(self, res=(192,192), show=False):
        x_render_gt = super().render((192,192), self.obs_list, show=False)
        x_render = cv2.resize(self.x_render.astype(np.uint8), (192,192), interpolation=cv2.INTER_NEAREST)
        x_render = cv2.hconcat([x_render, x_render_gt])
        ####
        m_val = torch.sigmoid(self.render_model.scene_cell)[0,:6000].detach().cpu().numpy()
        m_key = self.render_model.scene_code[0,:6000].detach().cpu().numpy()
        print(m_val.shape, m_key.shape)
        scale = 16
        mem_canvas = np.zeros((scale*np.abs(self.render_model.cell_range[0] - self.render_model.cell_range[1]), 
                            scale*np.abs(self.render_model.cell_range[2] - self.render_model.cell_range[3]), 3)).astype(np.uint8)
        #print(m_key.max(0), self.render_model.cell_range)
        for i in range(m_key.shape[0]):
            color = (int(255*m_val[i,0]), int(255*m_val[i,1]), int(255*m_val[i,2]))
            cv2.circle(mem_canvas, (int(scale*(m_key[i,0]-self.render_model.cell_range[0])), int(scale*(m_key[i,1]-self.render_model.cell_range[2]))), \
                        2, color, -1)
        mem_canvas = cv2.flip(mem_canvas, 0)
        mem_canvas = cv2.resize(mem_canvas, (192,192), interpolation=cv2.INTER_NEAREST)
        x_render = cv2.hconcat([x_render, mem_canvas])
        #print(mem_canvas.shape)
        ####
        if show:
            cv2.imshow("MazeSceneRepEnv", x_render)
            #cv2.imshow("Memory", mem_canvas)
        return x_render

if __name__ == "__main__":
    # Select maze type.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', nargs='?', type=str, default="MazeGridRandom", help='Maze type.')
    maze_type = parser.parse_args().type
    if maze_type == "MazeGridRoom":
        maze_obj = maze.MazeGridRoom()
    elif maze_type == "MazeGridRandom":
        maze_obj = maze.MazeGridRandom2(size=(11,11), room_total=5)
        #maze_obj = maze.MazeGridRandom2(size=(17,17), room_total=10)
        #maze_obj = maze.MazeGridRandom2(size=(51,51), room_total=50)
    elif maze_type == "MazeGridDungeon":
        maze_obj = maze.MazeGridDungeon()
    elif maze_type == "MazeBoardRoom":
        maze_obj = maze.MazeBoardRoom()
    elif maze_type == "MazeBoardRandom":
        maze_obj = maze.MazeBoardRandom()
    else:
        maze_obj = maze.MazeGridRandom()

    # Initial Env
    env = MazeSceneRepEnv(maze_obj, render_res=(64,64), exp_path='experiments/model_smn/')
    state, info = env.reset()
    env.render(show=True)

    while(True):
        # Control Handle
        k = cv2.waitKey(0)
        run_step = False
        # Exit
        if k == 27:
            break
        # Reset Maze
        if k == 13:
            state, info = env.reset()
            env.render(show=True)
            continue
        # Reset Pose
        if k == 32:
            state, info = env.reset(gen_maze=False)
            env.render(show=True)
            continue
        if k == ord('w'):
            action = 0
            run_step = True
        if k == ord('s'):
            action = 5
            run_step = True
        if k == ord('a'):
            action = 1
            run_step = True
        if k == ord('d'):
            action = 2
            run_step = True
        if k == ord('e'):
            action = -1
            add_obs = True
            run_step = True
            print("\nAdd Observation")
        else:
            add_obs = False
        
        if run_step:
            state_next, reward, done, info = env.step(action, add_obs=add_obs)
            print("\r", info["pose"], reward, done, end="\t")
            env.render(show=True)
            state = state_next.copy()

            if done:
                state, info = env.reset()
                env.render(show=True)
            
            