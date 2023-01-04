import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset_maze

############ Draw Results ############
def draw_query_maze(net, data_path, obs_size=3, row_size=32, gen_size=10, shuffle=False, border=[1,4,3]):
    img_list = []
    for it in range(gen_size):
        img_row = []
        x_obs, v_obs, x_query_gt, v_query = dataset_maze.get_file_batch(data_path, obs_size, row_size)
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

############ Evaluation ############
def eval_maze(net, data_path, obs_size=3, max_batch=1000, shuffle=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rmse_record = []
    mae_record = []
    ce_record = []
    for it in range(max_batch):
        img_row = []
        x_obs, v_obs, x_query_gt, v_query = dataset_maze.get_file_batch(data_path, obs_size, 32)
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
