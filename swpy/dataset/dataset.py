import re
from glob import glob
import numpy as np
import pandas as pd

import sys
sys.path.append('../utils/')

import torch.utils.data as data
from transform import standardize

class DstModelDataset(data.Dataset):
    def __init__(self, dst_flat_data, statistics_df=None):
        self.dst_flat_data = dst_flat_data
        self.statistics_df = statistics_df

    def __len__(self):
        return len(self.dst_flat_data)

    def __getitem__(self, index):
        d = self.dst_flat_data[index]
        dst_f      = d['DST_f']
        dst_p      = d['DST_p']
        dst_diff   = d['DST_diff']
        omni_data  = d['OMNI']

        # ============================
        # data の標準化
        # ============================
        if self.statistics_df is not None:
            omni_data_std = dict()
            mean = self.statistics_df['mean']['DST']
            var  = self.statistics_df['var']['DST']

            dst_f_std = standardize(dst_f, mean, var)
            dst_p_std = standardize(dst_p, mean, var)
            dst_diff_std = standardize(dst_diff, mean, var)

            for key, val in omni_data.items():
                mean = self.statistics_df['mean'][key]
                var  = self.statistics_df['var'][key]
                omni_data_std[key] = standardize(omni_data[key], mean, var)

            return dst_f_std, dst_p_std, dst_diff_std, omni_data_std
        return  dst_f, dst_p, dst_diff, omni_data



# class DSTFlatData():
#     def __init__(self, pft, udt, omni_size, omni_step_min, pq_inout_name_map, dst_data_dir, omni_data_dir):
#         self.pft               = pft
#         self.udt               = udt
#         self.omni_size         = omni_size
#         self.omni_step_min     = omni_step_min
#         self.pq_inout_name_map = pq_inout_name_map
#         self.dst_data_dir      = dst_data_dir
#         self.omni_data_dir     = omni_data_dir
#         self.dst_flat_data     = []


#     def has_omni_data_null(self, omni_data):
#         for key, nparray in omni_data.items():
#             if np.isnan(nparray).any():
#                 return True
#         return False


#     def _append_event_data(self, dst_data_path, omni_data_path):
#         dst_df = pd.read_csv(dst_data_path)
#         omni_df = pd.read_csv(omni_data_path)

#         dst_rows = len(dst_df)
#         for i in range(dst_rows - (self.udt + self.pft - 1)):
#             dst_step = i + self.udt
#             omni_data = dict()

#             for key, val in self.pq_inout_name_map.items():

#                 if 'DST' in key:
#                     dst_f = np.double(dst_df[dst_step + (self.pft - 1):dst_step + 1 + (self.pft - 1)][key])
#                     dst_p = dst_df[i:dst_step][key].values
#                     dst_diff = dst_f - dst_p[-1]

#                 else:
#                     omni_range_start = i * int(60 / self.omni_step_min)
#                     omni_range_end = i * int(60 / self.omni_step_min) + (self.udt - 1) * int(60 / self.omni_step_min) + 1
#                     this_df = omni_df[omni_range_start:omni_range_end]

#                     omni_data[val] = this_df[key].values

#             if self.has_omni_data_null(omni_data):
#                 break

#             self.dst_flat_data.append({'DST_f': dst_f, 'DST_p': dst_p, 'DST_diff': dst_diff, 'omni': omni_data})

#     def create_flat_data(self):
#         self.dst_flat_data = []
#         dst_data_paths  = sorted(glob(self.dst_data_dir + 'storm*.csv'))
#         omni_data_paths = sorted(glob(self.omni_data_dir + 'storm*.csv'))

#         for dst_data_path, omni_data_path in zip(dst_data_paths, omni_data_paths):
#             self._append_event_data(dst_data_path, omni_data_path)

#         return self.dst_flat_data



class DstTargetDataset():

    event_dt_prog = re.compile(r'\d{10}_\d{10}')

    def __init__(self, pft, udt, omni_size, omni_step_min, pq_inout_name_map, dst_data_dir, omni_data_dir):
        self.pft               = pft
        self.udt               = udt
        self.omni_size         = omni_size
        self.omni_step_min     = omni_step_min
        self.pq_inout_name_map = pq_inout_name_map
        self.dst_data_dir      = dst_data_dir
        self.omni_data_dir     = omni_data_dir
        self.events_dataset    = dict()
        self.flat_dataset      = dict()

    def create_events_dataset(self):
        dst_data_paths  = sorted(glob(self.dst_data_dir + 'storm*.csv'))
        omni_data_paths = sorted(glob(self.omni_data_dir + 'storm*.csv'))

        for dst_data_path, omni_data_path in zip(dst_data_paths, omni_data_paths):
            dst_filename = dst_data_path.split('/')[-1]
            event_dt = self.event_dt_prog.search(dst_filename).group()
            self.events_dataset[event_dt] = dict()
            evddt = self.events_dataset[event_dt]
            evddt['OMNI'] = dict()

            for key, val in self.pq_inout_name_map.items():
                if 'DST' in key:
                    dst_key = key
                    evddt['DST_f']    = np.empty((0, 1))
                    evddt['DST_diff'] = np.empty((0, 1))
                    evddt['DST_p']    = np.empty((0, self.udt))
                else:
                    evddt['OMNI'][val] = np.empty((0, self.omni_size))

            self._append_event_data(evddt, dst_key, dst_data_path, omni_data_path)

        return self.events_dataset


    def create_flat_dataset(self):
        flat_dataset = []

        for key, data in self.events_dataset.items():
            n_data = len(data['DST_f'])
            for i in range(n_data):
                dst_f    = data['DST_f'][i]
                dst_diff = data['DST_diff'][i]
                dst_p    = data['DST_p'][i]

                omni_data = dict()
                for o_key, o_data in data['OMNI'].items():
                    omni_data[o_key] = o_data[i]

                if self._has_omni_data_null(omni_data):
                    # nan を含むデータ除外
                    break

                flat_dataset.append({'DST_f': dst_f, 'DST_p': dst_p, 'DST_diff': dst_diff, 'OMNI': omni_data})

        self.flat_dataset = flat_dataset
        return self.flat_dataset

    def get_events_dataset(self):
        return self.events_dataset

    def get_flat_dataset(self):
        return self.flat_dataset

    def _has_omni_data_null(self, omni_data):
        for key, nparray in omni_data.items():
            if np.isnan(nparray).any():
                return True
        return False


    def _append_event_data(self, dataset, dst_key, dst_data_path, omni_data_path):
        dst_df = pd.read_csv(dst_data_path)
        omni_df = pd.read_csv(omni_data_path)

        dst_rows = len(dst_df)
        for i in range(dst_rows - (self.udt + self.pft - 1)):
            dst_step = i + self.udt
            dst_f = np.double(dst_df[dst_step + (self.pft - 1):dst_step + 1 + (self.pft - 1)][dst_key])
            dataset['DST_f'] = np.append(dataset['DST_f'], np.array([[dst_f]]), axis=0)

            dst_p = dst_df[i:dst_step][dst_key].values
            dataset['DST_p'] = np.append(dataset['DST_p'], np.array([dst_p]), axis=0)

            dst_diff = dst_f - dst_p[-1]
            dataset['DST_diff'] = np.append(dataset['DST_diff'], np.array([[dst_diff]]), axis=0)

            omni_range_start = i * int(60 / self.omni_step_min)
            omni_range_end = i * int(60 / self.omni_step_min) + (self.udt - 1) * int(60 / self.omni_step_min) + 1
            this_df = omni_df[omni_range_start:omni_range_end]

            for key, val in self.pq_inout_name_map.items():
                if 'DST' in key:
                    continue

                values = this_df[key].values
                dataset['OMNI'][val] = np.append(dataset['OMNI'][val], np.array([values]), axis=0)
