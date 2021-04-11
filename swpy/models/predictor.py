import torch
import torch.nn as nn

from fc_layer import FCLayer

import pdb

class MlpPredictor(nn.Module):

    h_dim_1 = 100
    h_dim_2 = 10

    def __init__(self, phase, len_dst_p, len_omni, num_omni_pqs):
        # super(MlpPredictor, self).__init__()
        super().__init__()

        self.phase = phase
        self.num_omni_pqs = num_omni_pqs

        self.dst_p_net = FCLayer(len_dst_p, self.h_dim_1)
        omni_nets = []
        for i in range(num_omni_pqs):
            omni_nets.append(FCLayer(len_omni, self.h_dim_1))
        self.omni_nets = nn.ModuleList(omni_nets)

        self.linear_1 = FCLayer(self.h_dim_1*(1 + num_omni_pqs), self.h_dim_1*(1 + num_omni_pqs))
        self.linear_2 = FCLayer(self.h_dim_1*(1 + num_omni_pqs), self.h_dim_2*(1 + num_omni_pqs))
        self.linear_3 = FCLayer(self.h_dim_2*(1 + num_omni_pqs), 1)

    def forward(self, dst_p, omni_data):
        h = self.dst_p_net(dst_p)

        for (key, val), omni_net in zip(omni_data.items(), self.omni_nets):
            h2 = omni_net(val.float())
            h = torch.cat([h, h2], axis=1)

        h3 = self.linear_1(h)
        h4 = self.linear_2(h3)
        out = self.linear_3(h4)

        return out


class MlpPredictorOmniMean(nn.Module):

    h_dim_1 = 100
    h_dim_2 = 10

    def __init__(self, phase, len_dst_p, num_omni_pqs):
        super().__init__()
        # super(MlpPredictorOmniMean, self).__init__()

        self.phase = phase
        self.num_omni_pqs = num_omni_pqs

        self.dst_p_net = FCLayer(len_dst_p, self.h_dim_1)
        omni_nets = []
        for i in range(num_omni_pqs):
            omni_nets.append(FCLayer(1, self.h_dim_1))
        self.omni_nets = nn.ModuleList(omni_nets)

        self.linear_1 = FCLayer(self.h_dim_1*(1 + num_omni_pqs), self.h_dim_1*(1 + num_omni_pqs))
        self.linear_2 = FCLayer(self.h_dim_1*(1 + num_omni_pqs), self.h_dim_2*(1 + num_omni_pqs))
        self.linear_3 = FCLayer(self.h_dim_2*(1 + num_omni_pqs), 1)

    def forward(self, dst_p, omni_data):
        h = self.dst_p_net(dst_p)

        for (key, val), omni_net in zip(omni_data.items(), self.omni_nets):
            # pdb.set_trace()
            h2 = omni_net(val.mean(axis=1).view(-1, 1).float())
            h = torch.cat([h, h2], axis=1)

        h3 = self.linear_1(h)
        h4 = self.linear_2(h3)
        out = self.linear_3(h4)

        return out
