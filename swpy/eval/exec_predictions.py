import sys
import numpy as np

import torch
import gpytorch

sys.path.append('../dataset')

from utils.transform import inverse_standardize
from data_processes import create_inputs_with_omni_mean

def predict_by_NN(net, eval_dataset, st):
    dst_predict = np.arange(0)
    dst_gt      = np.arange(0)

    for dst_f, dst_p, dst_diff, omni_data in eval_dataset:
        x = create_inputs_with_omni_mean(dst_p.float(), omni_data)
        outputs = net(x)
        # outputs = net(dst_p.float(), omni_data)

        out = outputs.detach().numpy()

        dst_predict = np.append(dst_predict, inverse_standardize(out, st['mean']['DST'], st['var']['DST']))
        dst_gt = np.append(dst_gt,inverse_standardize(dst_f, st['mean']['DST'], st['var']['DST']))

    return dst_predict, dst_gt


def predict_by_GP(gp_model, likelihood, eval_dataset, st):

    for i, (_, dst_p, _, omni_data) in enumerate(eval_dataset):
        input_x = create_inputs_with_omni_mean(dst_p, omni_data)
        if i == 0:
            input_x_all = input_x

        else:
            input_x_all = torch.cat([input_x_all, input_x], dim=0)

    test_x_all = input_x_all.view(input_x_all.shape[0], -1)

    gp_model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gp_out = gp_model(test_x_all)

    mean = gp_out.mean
    cr = gp_out.confidence_region()


    mean     = inverse_standardize(mean, st['mean']['DST'], st['var']['DST'])
    cr_lower = inverse_standardize(cr[0], st['mean']['DST'], st['var']['DST'])
    cr_upper = inverse_standardize(cr[1], st['mean']['DST'], st['var']['DST'])

    mean = mean.detach().numpy()[0]
    cr_lower = cr_lower.detach().numpy()[0]
    cr_upper = cr_upper.detach().numpy()[0]

    return mean, cr_lower, cr_upper