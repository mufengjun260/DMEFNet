import copy

import torch
from torch import nn
import torch.nn.functional as F


class EMGFlowNet(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, (1, 63), padding=(0, 31))
        self.bn1_1 = torch.nn.BatchNorm2d(5)
        self.depthConv1 = torch.nn.Conv2d(5, 5 * 2, (5, 1), groups=5, bias=False)

        self.lstm0 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm1 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm2 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm3 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm4 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)

        self.bn1_1 = torch.nn.BatchNorm2d(5)
        self.bn1_2 = torch.nn.BatchNorm2d(10)
        self.AvgPool1 = torch.nn.AvgPool2d(1, 4)

        self.conv2_1 = torch.nn.Conv2d(10, 10, (1, 15), groups=5, padding=(0, 7))
        self.conv2_2 = torch.nn.Conv2d(10, 10, 1)
        self.bn2 = torch.nn.BatchNorm2d(10)
        self.AvgPool2 = torch.nn.AvgPool2d(1, 8)

        self.l1 = torch.nn.Linear(100, 3)

        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1_avg = torch.nn.Conv1d(10, 10, 1, bias=False)
        self.fc2_avg = torch.nn.Conv1d(10, 10, 1, bias=False)

        self.avg_pool_1 = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool_1 = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1_avg_1 = torch.nn.Conv1d(5, 5, 1, bias=False)
        self.fc2_avg_1 = torch.nn.Conv1d(5, 5, 1, bias=False)

        self.param_avg = torch.nn.Parameter(torch.as_tensor(1.), True)
        self.param_max = torch.nn.Parameter(torch.as_tensor(1.), True)

    def forward(self, x):
        bs, tp_bs, channel, length, _ = x.shape
        output_feature = []
        ori_x = x.reshape(bs, tp_bs, 1, channel, length).type(torch.float)

        self.lstm0.flatten_parameters()
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
        self.lstm4.flatten_parameters()

        ori_x_tmp, (hn, cn) = self.lstm0(ori_x[:, :, 0, 0, :])
        catted_ori_x = ori_x_tmp[:, -1, :].unsqueeze(1)
        ori_x_tmp, (hn, cn) = self.lstm1(ori_x[:, :, 0, 1, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm2(ori_x[:, :, 0, 2, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm3(ori_x[:, :, 0, 3, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_x_tmp, (hn, cn) = self.lstm4(ori_x[:, :, 0, 4, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_x_tmp[:, -1, :].unsqueeze(1)], dim=1)

        ori_x = self.conv1(catted_ori_x.unsqueeze(1))
        ori_x = self.bn1_1(ori_x)
        ori_x = F.elu(ori_x)
        output_feature.append(ori_x.clone().detach())
        #
        ori_x = self.depthConv1(ori_x)
        ori_x = self.bn1_2(ori_x)
        ori_x = F.elu(ori_x)
        ori_x = self.AvgPool1(ori_x)
        output_feature.append(ori_x.clone().detach())
        ori_x = self.conv2_2(self.conv2_1(ori_x))
        ori_x = self.bn2(ori_x)
        ori_x = F.elu(ori_x)
        output_feature.append(ori_x.clone().detach())

        # channel attention
        avg_out = self.fc2_avg(F.relu(self.fc1_avg(self.avg_pool(ori_x.squeeze(2)))))
        max_out = self.fc2_avg(F.relu(self.fc1_avg(self.max_pool(ori_x.squeeze(2)))))
        out = self.param_avg * avg_out + self.param_max * max_out
        ori_x = ori_x * out.unsqueeze(3)
        output_feature.append(out.clone().detach())
        ori_x = F.elu(ori_x)
        ori_x = self.AvgPool2(ori_x)

        ori_x = ori_x.reshape(bs, -1)

        # output_feature.append(copy.deepcopy(ori_x))
        ori_x = F.softmax(self.l1(ori_x), dim=1)

        return ori_x, output_feature

    pass
