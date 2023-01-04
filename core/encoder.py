import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderNetworkLight(nn.Module):
    def __init__(self, ch=64, csize=128):
        super(EncoderNetworkLight, self).__init__()
        self.net = nn.Sequential(
                # (ch,64,64)
                Conv2d(3, ch, 5, stride=2),
                nn.LeakyReLU(),
                # (ch,32,32)
                Conv2d(ch, ch, 5, stride=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(ch),
                # (ch/2,32,32)
                Conv2d(ch, ch*2, 3, stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(ch*2),
                # (ch/2,16,16)
                Conv2d(ch*2, csize, 3, stride=1),
                # (tsize,16,16)
        )
        
    def forward(self, x):
        out = self.net(x)
        return out

class EncoderNetworkTower(nn.Module):
    def __init__(self, csize=64):
        super(EncoderNetworkTower, self).__init__()
        self.csize = 64
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, csize, kernel_size=1, stride=1)

    def forward(self, x):
        skip_in  = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        skip_in = F.relu(self.conv4(r)) + skip_out
        skip_out  = F.relu(self.conv5(skip_in))

        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))
        return r

class EncoderNetworkTowerBN(nn.Module):
    def __init__(self, csize=64):
        super(EncoderNetworkTowerBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(csize)
        self.conv8 = nn.Conv2d(256, csize, kernel_size=1, stride=1)

    def forward(self, x):
        skip_in  = F.relu(self.conv1(x))
        skip_out = self.bn2(F.relu(self.conv2(skip_in)))

        r = F.relu(self.conv3(skip_in))
        skip_in = self.bn4(F.relu(self.conv4(r))) + skip_out
        skip_out  = self.bn5(F.relu(self.conv5(skip_in)))

        r = self.bn6(F.relu(self.conv6(skip_in)))
        r = self.bn7(F.relu(self.conv7(r))) + skip_out
        r = F.relu(self.conv8(r))
        return r

from blurPooling import BlurPool2d
class EncoderNetworkRes(nn.Module):
    def __init__(self, ch=64, csize=128, down_size=4):
        super(EncoderNetworkRes, self).__init__()
        # Stem
        self.conv1 = Conv2d(3, ch, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(ch)
        #self.pool1 = BlurPool2d(filt_size=3, channels=ch, stride=2)
        # ResBlock 1
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = Conv2d(ch, ch, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(ch)
        self.conv3 = Conv2d(ch, ch, 3, stride=1)
        # Pool + Feature Dim
        #self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = BlurPool2d(filt_size=3, channels=ch, stride=2)
        self.conv4 = Conv2d(ch, ch*2, 1, stride=1)
        # ResBlock 2
        self.bn5 = nn.BatchNorm2d(ch*2)
        self.conv5 = Conv2d(ch*2, ch*2, 3, stride=1)
        self.bn6 = nn.BatchNorm2d(ch*2)
        self.conv6 = Conv2d(ch*2, ch*2, 3, stride=1)
        # Feature Dim
        self.bn7 = nn.BatchNorm2d(ch*2)
        self.conv7 = Conv2d(ch*2, csize, 1, stride=1)

    def forward(self, x):
        # Stem
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        #out = self.pool1(out)
        # ResBlock 1
        hidden = self.conv2(F.leaky_relu(self.bn2(out), inplace=True))
        out = self.conv3(F.leaky_relu(self.bn3(hidden), inplace=True)) + out
        # Pool + Feature Dim
        out = self.pool3(out)
        out = self.conv4(out)
        # ResBlock 2
        hidden = self.conv5(F.leaky_relu(self.bn5(out), inplace=True))
        out = self.conv6(F.leaky_relu(self.bn6(hidden), inplace=True)) + out
        # Feature Dim
        out = self.conv7(self.bn7(out))
        return out

######################################################
# For GQN
class GQNTower(nn.Module):
    def __init__(self, vsize=7, csize=256):
        super(GQNTower, self).__init__()
        self.vsize = vsize
        self.csize = csize
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256+vsize, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256+vsize, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, csize, kernel_size=1, stride=1)

    def forward(self, x, v):
        # Resisual connection
        skip_in  = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out

        # Broadcast
        v = v.view(v.size(0), self.vsize, 1, 1).repeat(1, 1, r.shape[2], r.shape[3])
        
        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out  = F.relu(self.conv5(skip_in))

        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))

        return r

class GQNTowerBN(nn.Module):
    def __init__(self, vsize=7, csize=256):
        super(GQNTowerBN, self).__init__()
        self.vsize = vsize
        self.csize = csize
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256+vsize, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256+vsize, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, csize, kernel_size=1, stride=1)

    def forward(self, x, v):
        # Resisual connection
        skip_in  = F.relu(self.conv1(x))
        skip_out = self.bn2(F.relu(self.conv2(skip_in)))

        r = self.bn3(F.relu(self.conv3(skip_in)))
        r = self.bn4(F.relu(self.conv4(r))) + skip_out

        # Broadcast
        v = v.view(v.size(0), vsize, 1, 1).repeat(1, 1, 16, 16)
        
        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out  = self.bn5(F.relu(self.conv5(skip_in)))

        r = self.bn6(F.relu(self.conv6(skip_in)))
        r = self.bn7(F.relu(self.conv7(r))) + skip_out
        r = F.relu(self.conv8(r))

        return r