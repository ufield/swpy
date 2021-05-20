import sys
import datetime

import numpy as np
import pandas as pd

import torch.utils.data as data

import pdb

sys.path.append('../dataset')
sys.path.append('../models')

from dataset import DstModelDataset

def create_eval_dataset_per_event(dataset_per_event, st):
    eval_dataset_per_event = dict()

    for key, event_data in dataset_per_event.items():
        flat_data = []
        # 推論の場合は、nanがあってもそのままにしておく
        n_data = len(event_data['DST_f'])
        for i in range(n_data):
            dst_f    = event_data['DST_f'][i]
            dst_diff = event_data['DST_diff'][i]
            dst_p    = event_data['DST_p'][i]

            omni_data = dict()
            for o_key, o_data in event_data['OMNI'].items():
                omni_data[o_key] = o_data[i]

            flat_data.append({'DST_f': dst_f, 'DST_p': dst_p, 'DST_diff': dst_diff, 'OMNI': omni_data})

        dst_model_dataset  = DstModelDataset(flat_data, st)
        dataloader         = data.DataLoader(dst_model_dataset, batch_size=100, shuffle=False)

        eval_dataset_per_event[key]= iter(dataloader)

    return eval_dataset_per_event

def _pad_nan(arr, pad_num):
    pad_arr = np.full(pad_num, np.nan)
    return np.concatenate([pad_arr, arr])


def create_pandas_dfs_for_result_plot(dst_csv_file_path, omni_csv_file_path, dst_pred, cr_lower, cr_upper):
    dst_df  = pd.read_csv(dst_csv_file_path)
    omni_df = pd.read_csv(omni_csv_file_path)

    dt_conv_dst  = lambda x: datetime.datetime.strptime(str(x), "%Y%m%d%H")
    dt_conv_omni = lambda x: datetime.datetime.strptime(str(x), "%Y%m%d%H%M")

    dst_df['Time']  = dst_df['Time'].map(dt_conv_dst)
    omni_df['Time'] = omni_df['Time'].map(dt_conv_omni)

    obs_len = len(dst_df)
    pad_num = obs_len - len(dst_pred)

    dst_pad = _pad_nan(dst_pred, pad_num)
    cr_lower_pad = _pad_nan(cr_lower, pad_num)
    cr_upper_pad = _pad_nan(cr_upper, pad_num)


    dst_df['DST_pred'] = dst_pad
    dst_df['cr_lower'] = cr_lower_pad
    dst_df['cr_upper'] = cr_upper_pad

    return dst_df, omni_df

