import torch
import torch.nn as nn


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