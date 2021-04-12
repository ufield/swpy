import numpy as np

import torch.utils.data as data
import sys

sys.path.append('../dataset')

from dataset import DstModelDataset
from utils.transform import standardize, inverse_standardize

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

def predict(net, eval_dataset, st):
    dst_predict = np.arange(0)
    dst_gt      = np.arange(0)

    for dst_f, dst_p, dst_diff, omni_data in eval_dataset:
        outputs = net(dst_p.float(), omni_data)

        out = outputs.detach().numpy()

        dst_predict = np.append(dst_predict, inverse_standardize(out, st['mean']['DST'], st['var']['DST']))
        dst_gt = np.append(dst_gt,inverse_standardize(dst_f, st['mean']['DST'], st['var']['DST']))

    return dst_predict, dst_gt