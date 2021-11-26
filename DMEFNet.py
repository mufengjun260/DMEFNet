import torch
from torch import nn
from dense_coattn_layer import DCNLayer
import torch.nn.functional as F


class EMDCANet(nn.Module):
    def __init__(self, length):
        super().__init__()
        """
        EEGNet of EEG1
        """
        self.conv1 = torch.nn.Conv2d(1, 8, (1, 125), padding=(0, 62))
        self.bn1_1 = torch.nn.BatchNorm2d(8)
        self.depthConv1 = torch.nn.Conv2d(8, 8 * 2, (32, 1), groups=8, bias=False)
        self.bn1_2 = torch.nn.BatchNorm2d(16)
        self.AvgPool1 = torch.nn.AvgPool2d(1, 25)

        self.conv2_1 = torch.nn.Conv2d(16, 16, (1, 15), groups=16, padding=(0, 33))
        self.conv2_2 = torch.nn.Conv2d(16, 16, 1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        # outsize 16*1*20
        self.AvgPool2 = torch.nn.AvgPool2d(1, 4)

        """
        EEGNet of EEG2
        """
        self.conv3 = torch.nn.Conv2d(1, 4, (1, 63), padding=(0, 31))
        self.bn3_1 = torch.nn.BatchNorm2d(4)
        self.depthConv2 = torch.nn.Conv2d(4, 4 * 2, (32, 1), groups=4, bias=False)
        self.bn3_2 = torch.nn.BatchNorm2d(8)
        self.AvgPool3 = torch.nn.AvgPool2d(1, 4)

        self.conv4_1 = torch.nn.Conv2d(8, 8, (1, 15), groups=8, padding=(0, 7))
        self.conv4_2 = torch.nn.Conv2d(8, 8, 1)
        self.bn4 = torch.nn.BatchNorm2d(8)
        # outsize 8*1*50
        self.AvgPool4 = torch.nn.AvgPool2d(1, 5)

        """
        MCSNet of sEMG
        """
        self.conv5 = torch.nn.Conv2d(1, 4, (1, 63), padding=(0, 31))
        self.bn5_1 = torch.nn.BatchNorm2d(4)
        self.depthConv5 = torch.nn.Conv2d(4, 4 * 2, (10, 1), groups=4, bias=False)

        self.lstm0 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm1 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm2 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm3 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)
        self.lstm4 = torch.nn.LSTM(300, 300, 1, batch_first=True, dropout=0.3)

        self.bn6_2 = torch.nn.BatchNorm2d(8)
        self.AvgPool6 = torch.nn.AvgPool2d(1, 3)

        self.conv7_1 = torch.nn.Conv2d(8, 8, (1, 15), groups=8, padding=(0, 7))
        self.conv7_2 = torch.nn.Conv2d(8, 8, 1)
        self.bn7 = torch.nn.BatchNorm2d(8)
        self.AvgPool7 = torch.nn.AvgPool2d(1, 2)

        self.l3 = torch.nn.Linear(1328, 3)
        """
        DCA of sEMG and EEG
        """
        self.dense_coattn = DCNLayer(50, 50, 5, 8, 5, 0.3)

    def forward(self, x_EEG1, x_EEG2, x_sEMG):
        bs_EEG1, channel_EEG1, length_EEG1 = x_EEG1.shape
        bs_EEG2, channel_EEG2, length_EEG2 = x_EEG2.shape

        # EEG1
        ori_EEG1 = x_EEG1.reshape(bs_EEG1, 1, channel_EEG1, length_EEG1).type(torch.float)

        ori_EEG1 = self.conv1(ori_EEG1)
        ori_EEG1 = self.bn1_1(ori_EEG1)
        ori_EEG1 = self.depthConv1(ori_EEG1)
        ori_EEG1 = self.bn1_2(ori_EEG1)
        ori_EEG1 = F.elu(ori_EEG1)
        ori_EEG1 = self.AvgPool1(ori_EEG1)

        ori_EEG1 = self.conv2_2(self.conv2_1(ori_EEG1))
        ori_EEG1 = self.bn2(ori_EEG1)
        ori_EEG1 = F.elu(ori_EEG1)
        ori_EEG1 = self.AvgPool2(ori_EEG1)

        EEG1_Feature = ori_EEG1.reshape(bs_EEG1, -1)

        # EEG2
        ori_EEG2 = x_EEG2.reshape(bs_EEG2, 1, channel_EEG2, length_EEG2).type(torch.float)

        ori_EEG2 = self.conv3(ori_EEG2)
        ori_EEG2 = self.bn3_1(ori_EEG2)
        ori_EEG2 = self.depthConv2(ori_EEG2)
        ori_EEG2 = self.bn3_2(ori_EEG2)
        ori_EEG2 = F.elu(ori_EEG2)
        ori_EEG2 = self.AvgPool3(ori_EEG2)

        ori_EEG2 = self.conv4_2(self.conv4_1(ori_EEG2))
        ori_EEG2 = self.bn4(ori_EEG2)
        ori_EEG2 = F.elu(ori_EEG2)
        ori_EEG2 = self.AvgPool4(ori_EEG2)
        # MT2
        EEG2_Feature = ori_EEG2

        # sEMG
        bs_sEMG, tp_bs_sEMG, channel_sEMG, length_sEMG, _ = x_sEMG.shape
        output_feature = []
        ori_sEMG = x_sEMG.reshape(bs_sEMG, tp_bs_sEMG, 1, channel_sEMG, length_sEMG).type(torch.float)

        self.lstm0.flatten_parameters()
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
        self.lstm4.flatten_parameters()

        ori_sEMG_tmp, (hn, cn) = self.lstm0(ori_sEMG[:, :, 0, 0, :])
        catted_ori_x = ori_sEMG_tmp[:, -1, :].unsqueeze(1)
        ori_sEMG_tmp, (hn, cn) = self.lstm1(ori_sEMG[:, :, 0, 1, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_sEMG_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_sEMG_tmp, (hn, cn) = self.lstm2(ori_sEMG[:, :, 0, 2, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_sEMG_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_sEMG_tmp, (hn, cn) = self.lstm3(ori_sEMG[:, :, 0, 3, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_sEMG_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_sEMG_tmp, (hn, cn) = self.lstm4(ori_sEMG[:, :, 0, 4, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_sEMG_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_sEMG_tmp, (hn, cn) = self.lstm0(ori_sEMG[:, :, 0, 5, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_sEMG_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_sEMG_tmp, (hn, cn) = self.lstm1(ori_sEMG[:, :, 0, 6, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_sEMG_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_sEMG_tmp, (hn, cn) = self.lstm2(ori_sEMG[:, :, 0, 7, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_sEMG_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_sEMG_tmp, (hn, cn) = self.lstm3(ori_sEMG[:, :, 0, 8, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_sEMG_tmp[:, -1, :].unsqueeze(1)], dim=1)
        ori_sEMG_tmp, (hn, cn) = self.lstm4(ori_sEMG[:, :, 0, 9, :])
        catted_ori_x = torch.cat([catted_ori_x, ori_sEMG_tmp[:, -1, :].unsqueeze(1)], dim=1)

        ori_sEMG = self.conv5(catted_ori_x.unsqueeze(1))
        ori_sEMG = self.bn5_1(ori_sEMG)
        ori_sEMG = F.elu(ori_sEMG)

        ori_sEMG = self.depthConv5(ori_sEMG)
        ori_sEMG = self.bn6_2(ori_sEMG)
        ori_sEMG = F.elu(ori_sEMG)
        ori_sEMG = self.AvgPool6(ori_sEMG)

        ori_sEMG = self.conv7_2(self.conv7_1(ori_sEMG))
        ori_sEMG = self.bn7(ori_sEMG)
        ori_sEMG = F.elu(ori_sEMG)

        ori_sEMG = F.elu(ori_sEMG)
        ori_sEMG = self.AvgPool7(ori_sEMG)
        # MT1
        sEMG_Feature = ori_sEMG
        # sEMG_Feature = ori_sEMG.reshape(bs_sEMG, -1)

        sEMG_Feature_co_att, EEG2_Feature_co_att = \
            self.dense_coattn(sEMG_Feature.reshape(sEMG_Feature.shape[0], 8, -1),
                                                                     EEG2_Feature.reshape(EEG2_Feature.shape[0], 8, -1),
                                                                     torch.ones(sEMG_Feature.shape[0], 8,
                                                                                device='cuda'),
                                                                     torch.ones(EEG2_Feature.shape[0], 8,
                                                                                device='cuda'))


        bs_sEMG_fusion_feature, _, _  = sEMG_Feature_co_att.shape
        bs_EEG2_fusion_feature, _, _ = EEG2_Feature_co_att.shape

        sEMG_fusion_feature = EEG2_Feature_co_att.reshape(bs_sEMG_fusion_feature, -1)
        EEG2_fusion_feature = EEG2_Feature_co_att.reshape(bs_EEG2_fusion_feature, -1)

        Fusion_Feature = torch.cat([sEMG_fusion_feature, EEG2_fusion_feature], dim=1)
        Fusion_Feature = torch.cat([Fusion_Feature, EEG1_Feature], dim=1)

        ori_x = F.softmax(self.l3(Fusion_Feature), dim=1)

        return ori_x
