import torch
import torch.nn as nn
import torch.utils.data as data

from dataset import DstModelDataset

def create_inputs_with_omni_mean(dst_p, omni_data, num_omni_pqs=4, avg_window=10):
    num_vals = num_omni_pqs + 1
    inputs = torch.zeros(dst_p.shape[0], dst_p.shape[1], num_vals)
    avgPool1d = nn.AvgPool1d(avg_window, stride=avg_window)

    inputs[:, :, 0] = dst_p
    ii = 1
    for key, val in omni_data.items():
        val_v = val.view(val.shape[0], 1, val.shape[1])
        val_a = avgPool1d(val_v)
        val_a = val_a.view(val_a.shape[0], -1)
        inputs[:, :, ii] = val_a
        ii += 1

    return inputs


def arrange_train_data_for_gp(train_dtd, st):
    '''
    Gaussian Process 用の学習データ調整。学習データ全体を一気に返す
    '''
    train_dtd.create_events_dataset()
    train_dst_flat_data = train_dtd.create_flat_dataset()

    train_dataset = DstModelDataset(train_dst_flat_data, st)
    train_dataloader = data.DataLoader(train_dataset, batch_size=len(train_dst_flat_data), shuffle=False)

    dst_f, dst_p, dst_diff, omni = next(iter(train_dataloader))
    train_y = dst_f
    train_y = train_y.squeeze()
    train_x = create_inputs_with_omni_mean(dst_p, omni)
    train_x = train_x.view(train_x.shape[0], -1)

    return train_x, train_y
