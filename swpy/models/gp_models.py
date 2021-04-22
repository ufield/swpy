import gpytorch
from mean_modules import DstNNMean

import pdb

class DstNNExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, net, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = DstNNMean(net)
        # TODO: NNKernel の導入
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=(1,)),
            batch_shape=(1,)
        )

    def forward(self, x):
        mean_x = self.mean_module(x.view(x.shape[0], 6, 5)).view(1, -1) # [n_points, 1] -> [1, n_points]
        covar_x = self.covar_module(x) # [n_points, udt, num_vals] -> [1, n_points, n_points]
        # pdb.set_trace()

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)