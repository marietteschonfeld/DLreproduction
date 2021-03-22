import torch

class LSTS(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(torch.SRU, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.init_params()

    def init_params(self, std=0.1):
        self.weight = std*torch.randn_like(self.weight)

    def forward(self, x):
        return

    def backward(self, x):
        return
