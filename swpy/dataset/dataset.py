import re
from glob import glob
import numpy as np
import pandas as pd

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
        dst_data_paths  = sorted(glob(self.dst_data_dir + '*'))
        omni_data_paths = sorted(glob(self.omni_data_dir + '*'))

        for dst_data_path, omni_data_path in zip(dst_data_paths, omni_data_paths):
            dst_filename = dst_data_path.split('/')[-1]
            event_dt = self.event_dt_prog.search(dst_filename).group()
            self.events_dataset[event_dt] = dict()
            evddt = self.events_dataset[event_dt]

            for key, val in self.pq_inout_name_map.items():
                if 'DST' in key:
                    dst_key = key
                    evddt['DST_f']    = np.empty((0, 1))
                    evddt['DST_diff'] = np.empty((0, 1))
                    evddt['DST_p']    = np.empty((0, self.udt))
                else:
                    evddt[val] = np.empty((0, self.omni_size))

            self._append_event_data(evddt, dst_key, dst_data_path, omni_data_path)

        return self.events_dataset


    def create_flat_dataset(self):
        dst_data_paths  = sorted(glob(self.dst_data_dir + '*'))
        omni_data_paths = sorted(glob(self.omni_data_dir + '*'))

        for key, val in self.pq_inout_name_map.items():
            if 'DST' in key:
                dst_key = key
                self.flat_dataset['DST_f']    = np.empty((0, 1))
                self.flat_dataset['DST_diff'] = np.empty((0, 1))
                self.flat_dataset['DST_p']    = np.empty((0, self.udt))
            else:
                self.flat_dataset[val] = np.empty((0, self.omni_size))

        for dst_data_path, omni_data_path in zip(dst_data_paths, omni_data_paths):
            self._append_event_data(self.flat_dataset, dst_key, dst_data_path, omni_data_path)

        return self.flat_dataset

    def get_events_dataset(self):
        return self.events_dataset

    def get_flat_dataset(self):
        return self.flat_dataset


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
                dataset[val] = np.append(dataset[val], np.array([values]), axis=0)
