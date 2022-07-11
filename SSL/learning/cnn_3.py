import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_3(nn.Module):
    def __init__(self, n_class):
        super(CNN_3, self).__init__()

        #nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        #input_tensor.shape = [batch_size,C,H,W]
        #in_channel = C, out_channel = 期望输出通道数
        #inception: in_planes_i+1 = (n1x1 + n3x3 + n5x5 + pool_planes)_i
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=11, stride=2, padding=0),
        #7
                                   #nn.BatchNorm2d(96),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.conv2 = nn.Sequential(nn.Conv2d(32, 128, kernel_size=5, stride=2, padding=0),
        #32_64
                                   #nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.conv3 = nn.Sequential(nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=0),
        #64_128
                                    #nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        
        # x0.shape: torch.Size([7, 32, 37, 40])
        #x1.shape: torch.Size([7, 47360])
        self.dropout = nn.Dropout(p=0.4)
       
        self.fc1 = nn.Linear(12544, 1028)
        self.bn1 = nn.BatchNorm1d(1028)
        self.fc2 = nn.Linear(1028,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, n_class)

        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #print("x0.shape:", x.shape)
        x = x.view(x.size()[0], -1)
        #print("x1.shape:", x.shape)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        output = self.fc3(x)
        return output

    #整个dataloader的数据concate起来一起判断
    def get_embeddings(self, loader, device):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                # aug_1, img, aug_2, mask, label = data
                img_mask = (data[1] + data[3]).to(device)
                x = self.forward(img_mask)
                pred = x.max(dim=1)[1]
                ret.append(pred.cpu().numpy())
                y.append(data[4].cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
