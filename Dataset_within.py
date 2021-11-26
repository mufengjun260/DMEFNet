import copy
import math

import numpy as np
import torch
import itertools
from torch.utils import data
import scipy.io as scio


class Dataset(data.Dataset):
    def __init__(self, clip_length, step, human, with_remix, get_first_part=False, with_extra_small=False,
                 first_is_train=True, first_part_rate=0, input_second_index=None, logger_setup=None):

        self.with_remix = with_remix
        self.get_first_part = get_first_part
        self.with_extra_small = with_extra_small
        self.second_index = []
        meta_sEMG = scio.loadmat('Dataset/sEMG.mat')
        meta_EEG1 = scio.loadmat('Dataset/EEG1.mat')
        meta_EEG2 = scio.loadmat('Dataset/EEG2.mat')
        name_index = meta_sEMG.get('sEMG')['name'][0]
        file_sEMG_index = meta_sEMG.get('sEMG')['file'][0]
        file_EEG1_index = meta_EEG1.get('EEG1')['file'][0]
        file_EEG2_index = meta_EEG2.get('EEG2')['file'][0]

        sEMG_train = [];
        EEG1_train = [];
        EEG2_train = []
        Y_sEMG_train = [];
        Y_EEG1_train = [];
        Y_EEG2_train = []
        for i in range(len(human)):
            name = name_index[human[i]][0]
            sEMG_file = file_sEMG_index[human[i]][0]
            EEG1_file = file_EEG1_index[human[i]][0]
            EEG2_file = file_EEG2_index[human[i]][0]

            sEMG_nam_index = sEMG_file['nam']
            sEMG_data_index = sEMG_file['data']
            EEG1_nam_index = EEG1_file['nam']
            EEG1_data_index = EEG1_file['data']
            EEG2_nam_index = EEG2_file['nam']
            EEG2_data_index = EEG2_file['data']
            for j in range(len(sEMG_nam_index)):
                sEMG_nam = sEMG_nam_index[j][0]
                sEMG_data = sEMG_data_index[j][0]

                EEG1_nam = EEG1_nam_index[j][0]
                EEG1_data = EEG1_data_index[j][0]

                EEG2_data = EEG2_data_index[j][0]

                sEMG_d_index = sEMG_data['d']
                sEMG_start_index = sEMG_data['start_index']

                EEG1_d_index = EEG1_data['d']
                EEG1_start_index = EEG1_data['start_index']

                EEG2_d_index = EEG2_data['d']
                EEG2_start_index = EEG2_data['start_index']

                split_num = int(first_part_rate * len(sEMG_d_index))
                if first_is_train:
                    split_num = math.ceil(first_part_rate * len(sEMG_d_index))

                total_list = np.arange(len(sEMG_d_index)).tolist()
                total_combines = list(itertools.combinations(total_list, split_num))[
                    np.random.randint(len(sEMG_d_index))]
                first_index = list(total_combines)
                second_index = list(set(total_list).difference(set(first_index)))
                self.second_index.append(copy.deepcopy(second_index))

                if input_second_index is not None:
                    second_index = input_second_index[j]

                if not self.get_first_part:
                    start = 0
                    end = len(sEMG_d_index) - split_num
                else:
                    start = 0
                    end = split_num
                    # end = int(first_part_rate * len(d_index))
                # for k in range(len(d_index) - 1):
                #
                for this_index in range(start, end):
                    # for this_index in range(1):
                    if self.get_first_part:
                        k = first_index[this_index]
                    else:
                        k = second_index[this_index]
                    sEMG_tmp_start_index = sEMG_start_index[k].item()
                    EEG1_tmp_start_index = EEG1_start_index[k].item()
                    EEG2_tmp_start_index = EEG2_start_index[k].item()

                    sEMG_data_matrix = sEMG_d_index[k][sEMG_tmp_start_index:]
                    EEG1_data_matrix = EEG1_d_index[k][EEG1_tmp_start_index:]
                    EEG2_data_matrix = EEG2_d_index[k][EEG2_tmp_start_index:]

                    data_temper = []
                    if self.with_extra_small:
                        sEMG_data_matrix = sEMG_data_matrix[:]
                        EEG1_data_matrix = EEG1_data_matrix[:]
                        EEG2_data_matrix = EEG2_data_matrix[:]
                    step_threshold = (sEMG_data_matrix.shape[0] - clip_length) / step + 1
                    if EEG1_nam == 'sit':
                        EEG1_train.append(EEG1_data_matrix[:, [0, 2, 4, 6, 3, 7, 8, 19, 20, 22, 23, 26, 24, 28]])
                        Y_EEG1_train.append([0])
                        EEG2_train.append(EEG2_data_matrix[:, [0, 2, 4, 6, 3, 7, 8, 19, 20, 22, 23, 26, 24, 28]])
                        Y_EEG2_train.append([0])
                    if EEG1_nam == 'stand':
                        EEG1_train.append(EEG1_data_matrix[:, [0, 2, 4, 6, 3, 7, 8, 19, 20, 22, 23, 26, 24, 28]])
                        Y_EEG1_train.append([1])
                        EEG2_train.append(EEG2_data_matrix[:, [0, 2, 4, 6, 3, 7, 8, 19, 20, 22, 23, 26, 24, 28]])
                        Y_EEG2_train.append([1])
                    if EEG1_nam == 'walk':
                        EEG1_train.append(EEG1_data_matrix[:, [0, 2, 4, 6, 3, 7, 8, 19, 20, 22, 23, 26, 24, 28]])
                        Y_EEG1_train.append([2])
                        EEG2_train.append(EEG2_data_matrix[:, [0, 2, 4, 6, 3, 7, 8, 19, 20, 22, 23, 26, 24, 28]])
                        Y_EEG2_train.append([2])

                    sEMG_train_start = len(sEMG_train)
                    for l in range(int(step_threshold)):
                        if sEMG_nam == 'sit':
                            data_clip = sEMG_data_matrix[l * step:clip_length + l * step, 0:5]
                            data_temper.append(data_clip)
                            if len(data_temper) == 13:
                                sEMG_train.append(copy.deepcopy(data_temper))
                                Y_sEMG_train.append([0])
                                data_temper.pop(0)

                        if sEMG_nam == 'stand':
                            data_clip = sEMG_data_matrix[l * step:clip_length + l * step, 0:5]
                            data_temper.append(data_clip)
                            if len(data_temper) == 13:
                                sEMG_train.append(copy.deepcopy(data_temper))
                                Y_sEMG_train.append([1])
                                data_temper.pop(0)

                        if sEMG_nam == 'walk':
                            data_clip = sEMG_data_matrix[l * step:clip_length + l * step, 0:5]
                            data_temper.append(data_clip)
                            if len(data_temper) == 13:
                                sEMG_train.append(copy.deepcopy(data_temper))
                                Y_sEMG_train.append([2])
                                data_temper.pop(0)

                    if logger_setup is not None:
                        logger_setup.info(
                            "Human {0} Motion {1} Seq {2} added from {3} to {4}".format(name, sEMG_nam, k,
                                                                                        sEMG_train_start,
                                                                                        len(sEMG_train) - 1))

        self.sEMG = torch.tensor(
            np.abs(np.array(sEMG_train).transpose([0, 1, 3, 2]).reshape(-1, 13, 5, clip_length, 1)),
            dtype=float)
        self.Y_sEMG = (torch.tensor(np.array([Y_sEMG_train])))[0]
        self.EEG1 = torch.tensor(
            np.abs(np.array(EEG1_train).transpose([0, 2, 1])),
            dtype=float)
        self.Y_EEG1 = (torch.tensor(np.array([Y_EEG1_train])))[0]
        self.EEG2 = torch.tensor(
            np.abs(np.array(EEG2_train).transpose([0, 2, 1])),
            dtype=float)
        self.Y_EEG2 = (torch.tensor(np.array([Y_EEG2_train])))[0]

    def __len__(self):
        return len(self.sEMG)

    def get_second_index(self):
        return None

    def __getitem__(self, item):
        if self.with_remix:
            length = self.__len__()
            rand_idx = torch.randint(0, length, [1])
            rand_ratio = torch.rand([1])
            rand_idx_noise = torch.randint(0, length, [1])
            rand_ratio_noise = torch.rand([1]) * 0.2

            sEMG_label = self.Y_sEMG[item]
            EEG1_label = self.Y_EEG1[item]
            EEG2_label = self.Y_EEG2[item]

            while self.Y_sEMG[rand_idx] != sEMG_label:
                rand_idx = torch.randint(0, length, [1])

            sEMG_merged = \
                ((self.sEMG[item] * rand_ratio + self.sEMG[rand_idx] * (1 - rand_ratio)) * (1 - rand_ratio_noise) +
                 self.sEMG[rand_idx_noise] * rand_ratio_noise)[0]
            EEG1_merged = \
                ((self.EEG1[item] * rand_ratio + self.EEG1[rand_idx] * (1 - rand_ratio)) * (1 - rand_ratio_noise) +
                 self.EEG1[rand_idx_noise] * rand_ratio_noise)[0]
            EEG2_merged = \
                ((self.EEG2[item] * rand_ratio + self.EEG2[rand_idx] * (1 - rand_ratio)) * (1 - rand_ratio_noise) +
                 self.EEG2[rand_idx_noise] * rand_ratio_noise)[0]
        else:
            sEMG_merged = self.sEMG[item]
            sEMG_label = self.Y_sEMG[item]
            EEG1_merged = self.EEG1[item]
            EEG1_label = self.Y_EEG1[item]
            EEG2_merged = self.EEG2[item]
            EEG2_label = self.Y_EEG2[item]
        return sEMG_merged, sEMG_label, EEG1_merged, EEG1_label, EEG2_merged, EEG2_label
