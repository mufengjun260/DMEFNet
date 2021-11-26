import copy
import math

import numpy as np
import torch
from torch.utils import data
import scipy.io as scio


class Dataset(data.Dataset):
    def __init__(self, clip_length, step, human, with_remix, get_first_part=False, with_extra_small=False,
                 first_is_train=True, first_part_rate=0):

        self.with_remix = with_remix
        self.get_first_part = get_first_part
        self.with_extra_small = with_extra_small
        meta_EEG1 = scio.loadmat('EEG1.mat')
        name_index = meta_EEG1.get('EEG1')['name'][0]
        file_EEG1_index = meta_EEG1.get('EEG1')['file'][0]

        EEG1_train = []
        Y_EEG1_train = []
        for i in range(len(human)):
            name = name_index[human[i]][0]
            EEG1_file = file_EEG1_index[human[i]][0]
            EEG1_nam_index = EEG1_file['nam']
            EEG1_data_index = EEG1_file['data']

            for j in range(len(EEG1_nam_index)):
                EEG1_nam = EEG1_nam_index[j][0]
                EEG1_data = EEG1_data_index[j][0]

                EEG1_d_index = EEG1_data['d']
                EEG1_start_index = EEG1_data['start_index']

                split_num = int(first_part_rate * len(EEG1_d_index))
                if first_is_train:
                    split_num = math.ceil(first_part_rate * len(EEG1_d_index))

                if not self.get_first_part:
                    start = split_num
                    end = len(EEG1_d_index)
                else:
                    start = 0
                    end = split_num

                for k in range(start, end):
                    EEG1_tmp_start_index = EEG1_start_index[k].item()

                    EEG1_data_matrix = EEG1_d_index[k][EEG1_tmp_start_index:]

                    data_temper = []
                    if self.with_extra_small:
                        EEG1_data_matrix = EEG1_data_matrix[:]

                    if self.with_extra_small:
                        EEG1_data_matrix = EEG1_data_matrix[:]

                    if EEG1_nam == 'sit':
                        EEG1_train.append(EEG1_data_matrix[:, [0, 2, 4, 6, 3, 7, 8, 19, 20, 22, 23, 26, 24, 28]])
                        Y_EEG1_train.append([0])

                    if EEG1_nam == 'stand':
                        EEG1_train.append(EEG1_data_matrix[:, [0, 2, 4, 6, 3, 7, 8, 19, 20, 22, 23, 26, 24, 28]])
                        Y_EEG1_train.append([1])

                    if EEG1_nam == 'walk':
                        EEG1_train.append(EEG1_data_matrix[:, [0, 2, 4, 6, 3, 7, 8, 19, 20, 22, 23, 26, 24, 28]])
                        Y_EEG1_train.append([2])

            print('aaaaaa')

        self.EEG1 = torch.tensor(
            np.abs(np.array(EEG1_train).transpose([0, 2, 1])),
            dtype=float)
        self.Y_EEG1 = (torch.tensor(np.array([Y_EEG1_train])))[0]

    def __getitem__(self, item):
        EEG1_merged = self.EEG1[item]
        EEG1_label = self.Y_EEG1[item]
        return EEG1_merged, EEG1_label

    def __len__(self):
        return len(self.EEG1)

    def get_second_index(self):
        return None

