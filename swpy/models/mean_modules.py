from gpytorch.means import Mean

class DstNNMean(Mean):
    def __init__(self, net):
        super().__init__()
        net.eval()
        net.requires_grad_(False)
        self.net = net

    def forward(self, x):
        return self.net(x)
