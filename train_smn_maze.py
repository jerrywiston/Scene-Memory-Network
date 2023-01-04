import numpy as np
import cv2
import os
import json
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import configparser
import config_handle
import utils
from core.smn import SMN
from maze3d.gen_maze_dataset import gen_dataset
from maze3d import maze
from maze3d import maze_env

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

########

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

########

def eval_maze(net, color, pose, obs_size=3, max_batch=1000, shuffle=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rmse_record = []
    mae_record = []
    ce_record = []
    for it in range(max_batch):
        img_row = []
        x_obs, v_obs, x_query_gt, v_query = get_batch(color, pose, obs_size, 32)
        img_size = (x_obs.shape[-2], x_obs.shape[-1])
        vsize = v_obs.shape[-1]
        with torch.no_grad():
            x_query_sample = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
            # rmse
            mse_batch = (x_query_sample*255 - x_query_gt*255)**2
            rmse_batch = torch.sqrt(mse_batch.mean([1,2,3])).cpu().numpy()
            rmse_record.append(rmse_batch)
            # mae
            mae_batch = torch.abs(x_query_sample*255 - x_query_gt*255)
            mae_batch = mae_batch.mean([1,2,3]).cpu().numpy()
            mae_record.append(mae_batch)
            # ce
            ce_batch = nn.BCELoss()(x_query_sample, x_query_gt)
            ce_batch = ce_batch.mean().cpu().numpy().reshape(1,1)
            ce_record.append(ce_batch)
        fill_size = len(str(max_batch))
        print("\rProgress: "+str(it+1).zfill(fill_size)+"/"+str(max_batch), end="")
    
    print("\nDone~~")
    rmse_record = np.concatenate(rmse_record, 0)
    rmse_mean = rmse_record.mean()
    rmse_std = rmse_record.std()
    mae_record = np.concatenate(mae_record, 0)
    mae_mean = mae_record.mean()
    mae_std = mae_record.std()
    ce_record = np.concatenate(ce_record, 0)
    ce_mean = ce_record.mean()
    ce_std = ce_record.std()
    return {"rmse":[float(rmse_mean), float(rmse_std)],
            "mae" :[float(mae_mean), float(mae_std)],
            "ce"  :[float(ce_mean), float(ce_std)]}

############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', nargs='?', type=str, default="maze" ,help='Experiment name.')
parser.add_argument('--config', nargs='?', type=str, default="./config.conf" ,help='Config filename.')
config_file = parser.parse_args().config
config = configparser.ConfigParser()
config.read(config_file)
args = config_handle.get_config_strgqn(config)
args.exp_name = parser.parse_args().exp_name
args.img_size = (args.v[0]*args.down_size, args.v[1]*args.down_size)

# Print Training Information
print("Configure File: %s"%(config_file))
print("Experiment Name: %s"%(args.exp_name))
print("Number of world cells: %d"%(args.w))
print("Size of view cells: " + str(args.v))
print("Number of concepts: %d"%(args.c))
print("Number of channels: %d"%(args.ch))
print("Downsampling size of view cell: %d"%(args.down_size))
print("Number of draw layers: %d"%(args.draw_layers))
print("Size of view pose: %d"%(args.vsize))
if args.share_core:
    print("Share core: True")
else:
    print("Share core: False")

############ Data Gen ############
maze_obj = maze.MazeGridRandom2(obj_prob=0.3)
env = maze_env.MazeBaseEnv(maze_obj, render_res=args.img_size, fov=80*np.pi/180)

############ Create Folder ############
now = datetime.datetime.now()
tinfo = "%d-%d-%d"%(now.year, now.month, now.day)
exp_path = "experiments/"
model_name = args.exp_name + "_w%d_c%d"%(args.w, args.c)
model_path = exp_path + tinfo + "_" + model_name + "/"

img_path = model_path + "img/"
save_path = model_path + "save/"
if not os.path.exists(img_path):
    os.makedirs(img_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save config file
with open(model_path + 'config.conf', 'w') as cfile:
    config.write(cfile)

############ Networks ############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SMN(n_wrd_cells=args.w, view_size=args.v, csize=args.c, ch=args.ch, vsize=args.vsize, \
    draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)
params = list(net.parameters())
opt = optim.Adam(params, lr=5e-5, betas=(0.5, 0.999))

# ------------ Loss Function ------------
if args.loss_type == "MSE":
    criterion = nn.MSELoss()
elif args.loss_type == "MAE":
    criterion = nn.L1Loss()
elif args.loss_type == "CE":
    creterion = nn.BCELoss()
else:
    criterion = nn.MSELoss()
    
# ------------ Prepare Variable ------------
img_path = model_path + "img/"
save_path = model_path + "save/"
train_record = {"loss_query":[], "lh_query":[], "kl_query":[]}
eval_record = []
best_eval = 999999
steps = 0
epochs = 0
eval_step = 5000
zfill_size = len(str(args.total_steps))
batch_size = 32
gen_data_size = 100
gen_dataset_iter = 1000
samp_field = 3.0

while(True):    
    if steps % gen_dataset_iter == 0:
        print("Generate Dataset ...")
        color_data, depth_data, pose_data = \
            gen_dataset(env, gen_data_size, samp_range=samp_field, samp_size=2*int(args.max_obs_size))
        color_data = color_data.astype(float) / 255.
        print("\nDone")

    # ------------ Get data (Random Observation) ------------
    obs_size = np.random.randint(1,args.max_obs_size)
    x_obs, v_obs, x_query_gt, v_query = get_batch(color_data, pose_data, obs_size, batch_size)

    # ------------ Forward ------------
    net.zero_grad()
    if args.stochastic_unit:
        x_query, kl_query = net(x_obs, v_obs, x_query_gt, v_query, n_obs=obs_size)
        lh_query = criterion(x_query, x_query_gt).mean()
        kl_query = torch.mean(torch.sum(kl_query, dim=[1,2,3]))
        loss_query = lh_query + args.kl_scale*kl_query
        rec = [float(loss_query.detach().cpu().numpy()), float(lh_query.detach().cpu().numpy()), float(kl_query.detach().cpu().numpy())]
    else:
        x_query = net.sample(x_obs, v_obs, v_query, n_obs=obs_size)
        loss_query = criterion(x_query, x_query_gt).mean()
        rec = [float(loss_query.detach().cpu().numpy()), float(loss_query.detach().cpu().numpy()), 0]
            
    # ------------ Train ------------
    loss_query.backward()
    opt.step()
    net.sample_wcode(net.n_wrd_cells)
    steps += 1

    # ------------ Print Result ------------
    if steps % 100 == 0:
        print("[Ep %s/%s] loss_q: %f| lh_q: %f| kl_q: %f"%(str(steps), str(args.total_steps), rec[0], rec[1], rec[2]))

    # ------------ Output Image ------------
    if steps % eval_step == 0:
        ##
        obs_size = 6
        gen_size = 5
        ##
        print("------------------------------")
        print("Generate Test Data ...")
        color_data_test, depth_data_test, pose_data_test = \
            gen_dataset(env, gen_size*3, samp_range=samp_field, samp_size=2*int(args.max_obs_size))
        color_data_test = color_data_test.astype(float) / 255.
        print("Done!!")
        # Train
        print("Generate image ...")
        fname = img_path+str(int(steps/eval_step)).zfill(4)+"_train.png"
        canvas = draw_query_maze(net, color_data, pose_data, obs_size=obs_size, row_size=5, gen_size=1, shuffle=True)[0]
        cv2.imwrite(fname, canvas)
        # Test
        print("Generate testing image ...")
        fname = img_path+str(int(steps/eval_step)).zfill(4)+"_test.png"
        canvas = draw_query_maze(net, color_data_test, pose_data_test, obs_size=obs_size, row_size=5, gen_size=1, shuffle=True)[0]
        cv2.imwrite(fname, canvas)

        # ------------ Training Record ------------
        train_record["loss_query"].append(rec[0])
        train_record["lh_query"].append(rec[1])
        train_record["kl_query"].append(rec[2])
        print("Dump training record ...")
        with open(model_path+'train_record.json', 'w') as file:
            json.dump(train_record, file)

        # ------------ Evaluation Record ------------
        #print("Evaluate Training Data ...")
        #eval_results_train = eval_maze(net, color_data, pose_data, obs_size=6, max_batch=400, shuffle=False)
        print("Evaluate Testing Data ...")
        eval_results_test = eval_maze(net, color_data_test, pose_data_test, obs_size=6, max_batch=400, shuffle=False)
        #eval_record.append({"steps":steps, "train":eval_results_train, "test":eval_results_test})
        eval_record.append({"steps":steps, "test":eval_results_test})
        print("Dump evaluation record ...")
        with open(model_path+'eval_record.json', 'w') as file:
            json.dump(eval_record, file)

        # ------------ Save Model ------------
        if steps%100000 == 0:
            print("Save model ...")
            torch.save(net.state_dict(), save_path + "model_" + str(steps).zfill(zfill_size) + ".pth")
                
        # Apply RMSE as the metric for model selection.
        if eval_results_test["rmse"][0] < best_eval:
            best_eval = eval_results_test["rmse"][0]
            print("Save best model ...")
            torch.save(net.state_dict(), save_path + "model.pth")
        print("Best Test RMSE:", best_eval)
        print("------------------------------")

    if steps >= args.total_steps:
        print("Save final model ...")
        torch.save(net.state_dict(), save_path + "model_" + str(steps).zfill(zfill_size) + ".pth")
        break