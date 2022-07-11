import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()

        # Inception(192, 64, 96, 128, 16, 32, 32)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1, stride=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1, stride=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(inplace=True),
            #nn.Conv2d(n3x3red, n3x3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(n3x3red, n3x3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, stride=1, padding=1),
            #nn.Conv2d(n5x5red, n5x5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            #把padding改成0，原来是1
            nn.Conv2d(in_planes, pool_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        #print x1.shape, x2.shape, x3.shape, x4.shape
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, drop_ratio, n_class):
        super(GoogLeNet, self).__init__()

        #nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        #input_tensor.shape = [batch_size,C,H,W]
        #in_channel = C, out_channel = 期望输出通道数
        #inception: in_planes_i+1 = (n1x1 + n3x3 + n5x5 + pool_planes)_i
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(192),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.inception_3 = nn.Sequential(Inception(192, 64, 96, 128, 16, 32, 32),
                                         Inception(256, 128, 128, 192, 32, 96, 64),
                                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.inception_4 = nn.Sequential(Inception(480, 192, 96, 208, 16, 48, 64),
                                         Inception(512, 160, 112, 224, 24, 64, 64),
                                         Inception(512, 128, 128, 256, 24, 64, 64),
                                         Inception(512, 112, 144, 288, 32, 64, 64),
                                         Inception(528, 256, 160, 320, 32, 128, 128),
                                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.inception_5 = nn.Sequential(Inception(832, 256, 160, 320, 32, 128, 128),
                                         Inception(832, 384, 192, 384, 48, 128, 128))

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=drop_ratio)
        self.fc1 = nn.Linear(1024, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception_3(x)
        x = self.inception_4(x)
        x = self.inception_5(x)
        x = self.avg_pool(x)
        #x = torch.squeeze(x, dim=-1)
        ###### 为什么要压缩2次？#########3
        #x = torch.squeeze(x, dim=-1)
        #### 源代码先dropout再linear，为什么？
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
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
