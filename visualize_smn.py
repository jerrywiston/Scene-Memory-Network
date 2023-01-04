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

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs='?', type=str, default="experiments/model_smn/" ,help='Experiment name.')
path = parser.parse_args().path
config = configparser.ConfigParser()
config.read(os.path.join(path, "config.conf"))
args = config_handle.get_config_strgqn(config)

############ Networks ############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SMN(n_wrd_cells=args.w, view_size=args.v, csize=args.c, ch=args.ch, vsize=args.vsize, \
    draw_layers=args.draw_layers, down_size=args.down_size, share_core=args.share_core).to(device)
net.load_state_dict(torch.load(path + "save/model.pth"))

##################################
scale=[3.0, 3.0, 3.0]
#scale=[10.0, 10.0, 3.0]
samp_size = 4096
net.strn.n_wrd_cells = samp_size
wcode = torch.rand(samp_size, 2).to(device)
wcode_temp = torch.ones((samp_size, 1)).to(device)*0.3
wcode = torch.cat([wcode, wcode_temp], 1)
#wcode = torch.rand(samp_size, 3).to(device)

with torch.no_grad():
    wcode[:,0] = (wcode[:,0] * 2 - 1) * scale[0]
    wcode[:,1] = (wcode[:,1] * 2 - 1) * scale[1]
    wcode[:,2] = (wcode[:,2] * 2 - 1) * scale[2]

v = torch.tensor([0,0,0,1,0,0,0], dtype=wcode.dtype).reshape(1,-1).to(device)
wcode_batch_trans = net.strn.wcode2cam(v, wcode)
relation, activation = net.strn.transform(wcode_batch_trans, view_size=(16,16))
#relation, activation = net.strn.transform(wcode.unsqueeze(0), view_size=(16,16))

relation = relation[0].detach().cpu().numpy()
activation = activation[0].detach().cpu().numpy()
wcode = wcode.detach().cpu().numpy()

cmap = plt.cm.get_cmap("jet")
# plot memory mask
plt.figure(figsize=(4,4))
plt.scatter(wcode[:,0], wcode[:,1], c=activation[:,0], cmap=cmap)

# plot routing (width)
temp1 = (np.argmax(relation, 1).astype(int) / 16).astype(int)
temp2 = np.argmax(relation, 1) % 16
plt.figure(figsize=(4,4))
plt.scatter(wcode[:,0], wcode[:,1], c=temp2/16, cmap=cmap)

# plot routing (height)
temp1 = (np.argmax(relation, 1).astype(int) / 16).astype(int)
temp2 = np.argmax(relation, 1) % 16
#plt.figure(figsize=(4,4))
#plt.scatter(wcode[:,0], wcode[:,1], c=temp1/16, cmap=cmap)

plt.show()