import torch.utils.data as data
import sys

sys.path.append('../models/')

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