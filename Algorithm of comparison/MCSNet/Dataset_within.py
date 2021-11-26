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
        meta_sEMG = scio.loadmat('sEMG.mat')
        name_index = meta_sEMG.get('sEMG')['name'][0]
        file_sEMG_index = meta_sEMG.get('sEMG')['file'][0]

        sEMG_train = []
        Y_sEMG_train = []
        for i in range(len(human)):
            name = name_index[human[i]][0]
            sEMG_file = file_sEMG_index[human[i]][0]

            sEMG_nam_index = sEMG_file['nam']
            sEMG_data_index = sEMG_file['data']

            for j in range(len(sEMG_nam_index)):
                sEMG_nam = sEMG_nam_index[j][0]
                sEMG_data = sEMG_data_index[j][0]

                sEMG_d_index = sEMG_data['d']
                sEMG_start_index = sEMG_data['start_index']

                split_num = int(first_part_rate * len(sEMG_d_index))
                if first_is_train:
                    split_num = math.ceil(first_part_rate * len(sEMG_d_index))

                total_list = np.arange(len(sEMG_d_index)).tolist()
                total_combines = list(itertools.combinations(total_list, split_num))[np.random.randint(len(sEMG_d_index))]
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

                for this_index in range(start, end):
                    if self.get_first_part:
                        k = first_index[this_index]
                    else:
                        k = second_index[this_index]
                    sEMG_tmp_start_index = sEMG_start_index[k].item()
                    sEMG_data_matrix = sEMG_d_index[k][sEMG_tmp_start_index:]

                    data_temper = []
                    if self.with_extra_small:
                        sEMG_data_matrix = sEMG_data_matrix[:]

                    step_threshold = (sEMG_data_matrix.shape[0] - clip_length) / step + 1

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
                            "Human {0} Motion {1} Seq {2} added from {3} to {4}".format(name, sEMG_nam, k, sEMG_train_start,
                                                                                        len(sEMG_train) - 1))

        self.sEMG = torch.tensor(
            np.abs(np.array(sEMG_train).transpose([0, 1, 3, 2]).reshape(-1, 13, 5, clip_length, 1)),
            dtype=float)
        self.Y_sEMG = (torch.tensor(np.array([Y_sEMG_train])))[0]

    def __len__(self):
        return len(self.sEMG)

    def get_second_index(self):
        return None

    def __getitem__(self, item):
        sEMG_merged = self.sEMG[item]
        sEMG_label = self.Y_sEMG[item]
        return sEMG_merged, sEMG_label

