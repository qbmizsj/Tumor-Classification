import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AlexNet(nn.Module):
    def __init__(self, n_class):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=0),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0),
                                   nn.BatchNorm2d(384),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
                                   nn.BatchNorm2d(384),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=0),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        #x0.shape: torch.Size([7, 256, 14, 15])
        #x1.shape: torch.Size([7, 53760])
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(53760, 16448)
        self.bn1 = nn.BatchNorm1d(16448)
        self.fc2 = nn.Linear(16448,1028)
        self.bn2 = nn.BatchNorm1d(1028)
        self.fc3 = nn.Linear(1028, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256,64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        print("x0.shape:", x.shape)
        x = x.view(x.size()[0], -1)
        print("x1.shape:", x.shape)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.dropout(x)
        x = self.bn4(F.relu(self.fc4(x)))
        x = self.dropout(x)
        output = self.fc5(x)
        return output

    # 整个dataloader的数据concate起来一起判断
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
