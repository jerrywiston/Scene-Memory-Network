import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        conv_out = self.conv(obs)
        advantage = self.fc_advantage(conv_out)
        value = self.fc_value(conv_out)
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q

class PolicyNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        conv_out = self.conv(obs)
        digit = self.fc(conv_out)
        prob = torch.softmax(digit, 1)
        return prob

###########################################################
from tsm import SpatialBlock, TSMBlock
class QNetSB(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetSB, self).__init__()
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

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        conv_out = self.conv(obs)
        advantage = self.fc_advantage(conv_out)
        value = self.fc_value(conv_out)
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q

class PolicyNetSB(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNetSB, self).__init__()
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

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        conv_out = self.conv(obs)
        digit = self.fc(conv_out)
        prob = torch.softmax(digit, 1)
        return prob

###########################################################
from tsm import SpatialBlock, TSMBlock
class QNetTSM(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetTSM, self).__init__()
        self.conv = nn.Sequential(
            SpatialBlock(frame_dim=3, out_dim=32, kernal_size=7, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=32, shift_dim=4, kernal_size=5, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=64, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        conv_out = self.conv(obs)
        advantage = self.fc_advantage(conv_out)
        value = self.fc_value(conv_out)
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q

class PolicyNetTSM(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNetTSM, self).__init__()
        self.conv = nn.Sequential(
            SpatialBlock(frame_dim=3, out_dim=32, kernal_size=7, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=32, shift_dim=4, kernal_size=5, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=64, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        conv_out = self.conv(obs)
        digit = self.fc(conv_out)
        prob = torch.softmax(digit, 1)
        return prob

###########################################################
from cell_net import CellNet
class QNetCellTSM(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetCellTSM, self).__init__()
        self.cnet = CellNet(32, 256, 16)
        self.conv = nn.Sequential(
            SpatialBlock(frame_dim=3, out_dim=32, kernal_size=7, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=32, shift_dim=4, kernal_size=5, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=64, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size+256, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size+256, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        cell_feature = self.cnet(s["cell_s2"], s["code_s2"])
        obs = s["obs"]
        conv_out = self.conv(obs)
        advantage = self.fc_advantage(torch.cat([conv_out, cell_feature],1))
        value = self.fc_value(torch.cat([conv_out, cell_feature],1))
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q

class PolicyNetCellTSM(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNetCellTSM, self).__init__()
        self.cnet = CellNet(32, 256, 16)
        self.conv = nn.Sequential(
            SpatialBlock(frame_dim=3, out_dim=32, kernal_size=7, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=32, shift_dim=4, kernal_size=5, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=64, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size+256, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        cell_feature = self.cnet(s["cell_s2"], s["code_s2"])
        obs = s["obs"]
        conv_out = self.conv(obs)
        digit = self.fc(torch.cat([conv_out, cell_feature],1))
        prob = torch.softmax(digit, 1)
        return prob

###########################################################
class QNetMultiCellTSM(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetMultiCellTSM, self).__init__()
        self.cnet1 = CellNet(32, 256, 16)
        self.cnet2 = CellNet(32, 256, 16)
        self.cnet3 = CellNet(32, 256, 16)

        self.conv = nn.Sequential(
            SpatialBlock(frame_dim=3, out_dim=32, kernal_size=7, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=32, shift_dim=4, kernal_size=5, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=64, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size+256*3, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size+256*3, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        cell_s1_feature = self.cnet1(s["cell_s1"], s["code_s1"])
        cell_s2_feature = self.cnet2(s["cell_s2"], s["code_s2"])
        cell_s3_feature = self.cnet3(s["cell_s3"], s["code_s3"])
        obs = s["obs"]
        conv_out = self.conv(obs)
        advantage = self.fc_advantage(torch.cat([conv_out, cell3_feature, cell5_feature, cell7_feature],1))
        value = self.fc_value(torch.cat([conv_out, cell3_feature, cell5_feature, cell7_feature],1))
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q

class PolicyNetMultiCellTSM(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNetMultiCellTSM, self).__init__()
        self.cnet1 = CellNet(32, 256, 16)
        self.cnet2 = CellNet(32, 256, 16)
        self.cnet3 = CellNet(32, 256, 16)

        self.conv = nn.Sequential(
            SpatialBlock(frame_dim=3, out_dim=32, kernal_size=7, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=32, shift_dim=4, kernal_size=5, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=32, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
            TSMBlock(frame_dim=64, out_dim=64, shift_dim=8, kernal_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size+256*3, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        cell_s1_feature = self.cnet1(s["cell_s1"], s["code_s1"])
        cell_s2_feature = self.cnet2(s["cell_s2"], s["code_s2"])
        cell_s3_feature = self.cnet3(s["cell_s3"], s["code_s3"])
        obs = s["obs"]
        conv_out = self.conv(obs)
        digit = self.fc(torch.cat([conv_out, cell_s1_feature, cell_s2_feature, cell_s3_feature],1))
        prob = torch.softmax(digit, 1)
        return prob

###########################################################
class QNetAction(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetAction, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size + n_actions*4, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size + n_actions*4, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        a_rec = s["a_rec"]
        conv_out = self.conv(obs)
        feature = torch.cat([conv_out, a_rec], 1)
        advantage = self.fc_advantage(feature)
        value = self.fc_value(feature)
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q

class PolicyNetAction(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNetAction, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + n_actions*4, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        a_rec = s["a_rec"]
        conv_out = self.conv(obs)
        feature = feature = torch.cat([conv_out, a_rec], 1)
        digit = self.fc(feature)
        prob = torch.softmax(digit, 1)
        return prob

########################################################### 
class QNetCell(nn.Module):
    def __init__(self, input_shape, n_actions, cell_dim):
        super(QNetCell, self).__init__()
        self.query_var = torch.nn.Linear(1, 256, bias=False)
        self.proj_cell = torch.nn.Conv1d(cell_dim, 256, 1)
        self.proj_code = torch.nn.Conv1d(7, 256, 1)
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=256, num_heads=16)

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size + 256, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        cell = self.proj_cell(s["cell"].permute(0,2,1))
        cell = F.relu(cell).permute(2,0,1)
        code = self.proj_code(s["code"].permute(0,2,1))
        code = F.relu(code).permute(2,0,1)
        query = self.query_var(torch.ones(obs.shape[0],1).to(obs.get_device())).unsqueeze(0)
        cell_fusion, _ = self.multihead_attn(query, code, cell)
        cell_fusion = cell_fusion.view(obs.shape[0], cell_fusion.shape[-1])
        conv_out = self.conv(obs)
        advantage = self.fc_advantage(torch.cat([conv_out, cell_fusion],1))
        value = self.fc_value(torch.cat([conv_out, cell_fusion],1))
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q

class PolicyNetCell(nn.Module):
    def __init__(self, input_shape, n_actions, cell_dim):
        super(PolicyNetCell, self).__init__()
        self.query_var = torch.nn.Linear(1, 256, bias=False)
        self.proj_cell = torch.nn.Conv1d(cell_dim, 256, 1)
        self.proj_code = torch.nn.Conv1d(7, 256, 1)
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=256, num_heads=16)

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + 256, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        cell = self.proj_cell(s["cell"].permute(0,2,1))
        cell = F.relu(cell).permute(2,0,1)
        code = self.proj_code(s["code"].permute(0,2,1))
        code = F.relu(code).permute(2,0,1)
        query = self.query_var(torch.ones(obs.shape[0],1).to(obs.get_device())).unsqueeze(0)
        cell_fusion, _ = self.multihead_attn(query, code, cell)
        cell_fusion = cell_fusion.view(obs.shape[0], cell_fusion.shape[-1])
        conv_out = self.conv(obs)
        digit = self.fc(torch.cat([conv_out, cell_fusion],1))
        prob = torch.softmax(digit, 1)
        return prob