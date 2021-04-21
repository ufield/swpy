import gpytorch
from mean_modules import DstNNMean

import pdb

class DstNNExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, net, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = DstNNMean(net)
        # TODO: NNKernel の導入
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x).squeeze() # [batch_size, 1] -> [batch_size]
        covar_x = self.covar_module(x.view(x.shape[0], -1)) # [batch_size, udt, num_vals] -> [batch_size, udt x num_vals]
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
