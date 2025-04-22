import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
EPS = 1e-10


def he_init_weights(module):
    """
    Initialize network weights using the He (Kaiming) initialization strategy.

    :param module: Network module
    :type module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)


class DDC(nn.Module):
    def __init__(self, input_dim, n_clusters):
        """
        DDC clustering module

        :param input_dim: Shape of inputs.
        :param cfg: DDC config. See `config.defaults.DDC`
        """
        super().__init__()
        hidden_layers = [nn.Linear(input_dim[0], 100), nn.ReLU(), nn.BatchNorm1d(num_features=100)]
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(100, n_clusters), nn.Softmax(dim=1))

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden

class DDC2(nn.Module):
    def __init__(self, input_dim, n_clusters):
        """
        DDC clustering module

        :param input_dim: Shape of inputs.
        :param cfg: DDC config. See `config.defaults.DDC`
        """
        super().__init__()
        hidden_layers = [nn.Linear(input_dim[0], 100), nn.ReLU(), nn.BatchNorm1d(num_features=100)]
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(100, n_clusters), nn.BatchNorm1d(num_features=n_clusters))

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden


class WeightedMean(nn.Module):
    """
    Weighted mean fusion.

    :param cfg: Fusion config. See config.defaults.Fusion
    :param input_sizes: Input shapes
    """
    def __init__(self, n_views, input_sizes):
        super().__init__()
        self.n_views = n_views
        self.weights = nn.Parameter(torch.full((self.n_views,), 1 / self.n_views), requires_grad=True)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def get_weighted_sum_output_size(self, input_sizes):
        flat_sizes = [np.prod(s) for s in input_sizes]
        return [flat_sizes[0]]

    def forward(self, inputs):
        return _weighted_sum(inputs, self.weights, normalize_weights=True)
class link_embdeing(nn.Module):
    def __init__(self, n_views, input_size):
        self.n_views = n_views
        self.linear = nn.Linear(input_size*n_views, input_size)

    def forward(self, inputs):
        features = inputs.flatten()
        out = self.linear(features)
        return out



class WeightedMean2(nn.Module):
    """
    Weighted mean fusion.

    :param cfg: Fusion config. See config.defaults.Fusion
    :param input_sizes: Input shapes
    """
    def __init__(self, n_views, input_sizes):
        super().__init__()
        self.n_views = n_views
        #self.weights = nn.Parameter(torch.full((self.n_views,), 1 / self.n_views), requires_grad=True)
        self.weights = nn.Parameter(torch.tensor([0.3, 0.3, 0.4]), requires_grad=True)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def get_weighted_sum_output_size(self, input_sizes):
        flat_sizes = [np.prod(s) for s in input_sizes]
        return [flat_sizes[0]]

    def forward(self, inputs):
        return _weighted_sum(inputs, self.weights, normalize_weights=True)

def _weighted_sum(tensors, weights, normalize_weights=True):
    if normalize_weights:
        weights = F.softmax(weights, dim=0)
    out = torch.sum(weights[None, None, :] * torch.stack(tensors, dim=-1), dim=-1)
    return out


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, feature_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, kernel_size=5, out_channels=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, kernel_size=5, out_channels=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, kernel_size=3, out_channels=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, kernel_size=3, out_channels=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.layer(x)


class BaseMVC(nn.Module):
    def __init__(self, input_size, feature_dim, class_num):
        super(BaseMVC, self).__init__()
        self.encoder = Encoder(input_size, feature_dim)
        self.cluster_module = DDC2([feature_dim], class_num)
        self.apply(he_init_weights)

    def forward(self, x):
        z = self.encoder(x)
        output, hidden = self.cluster_module(z)
        return output


class SiMVC(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num):
        super(SiMVC, self).__init__()
        self.encoders = []
        self.view = view
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        input_sizes = []
        for _ in range(view):
            input_sizes.append([feature_dim])
        self.fusion_module = WeightedMean(view, input_sizes=input_sizes)
        #self.fusion_module = link_embdeing(view, input_size=feature_dim)
        self.cluster_module = DDC2([feature_dim], class_num)

    def forward(self, xs):
        zs = []
        for v in range(self.view):
            zs.append(self.encoders[v](xs[v]))
        fused = self.fusion_module(zs)
        fused, hidden = self.cluster_module(fused)
        return zs, fused


class MVC(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num):
        super(MVC, self).__init__()
        self.encoders = []
        self.view = view
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        input_sizes = []
        for _ in range(view):
            input_sizes.append([feature_dim])
        self.fusion_module = WeightedMean2(view, input_sizes=input_sizes)
        self.cluster_module = DDC(self.fusion_module.output_size, class_num)
        self.apply(he_init_weights)

    def forward(self, xs):
        zs = []
        for v in range(self.view):
            zs.append(self.encoders[v](xs[v]))
        fused = self.fusion_module(zs)
        output, hidden = self.cluster_module(fused)
        return output, hidden




class DE_MCC(nn.Module):
    def __init__(self, view_old, view_new, input_size, feature_dim, class_num):
        super(DE_MCC, self).__init__()
        self.view = view_new
        self.old_model = SiMVC(view_old, input_size, feature_dim, class_num)
        self.new_model = SiMVC(view_new, input_size, feature_dim, class_num)
        self.single = BaseMVC(input_size[view_new-1], feature_dim, class_num)
        # self.gate = WeightedMean(3, [[feature_dim], [feature_dim], [feature_dim]])
        # self.gate = DDC([class_num*3], class_num)
        self.gate = nn.Sequential(nn.Linear(3*class_num, class_num), nn.Softmax(dim=1))

        self.cluster_module = DDC([feature_dim], class_num)
        self.apply(he_init_weights)

    def forward(self, xs):
        zs_old, fused_old = self.old_model(xs)
        zs_new, fused_new = self.new_model(xs)
        single = self.single(xs[self.view-1])
        fuse = [fused_new, fused_old, single]
        hidden = torch.cat(fuse, dim=1)
        output = self.gate(hidden)
        return zs_old, zs_new, output, hidden, fuse

    # def forward(self, xs):
    #     #zs_old, fused_old = self.old_model(xs)
    #     #zs_new, fused_new = self.new_model(xs)
    #     single = self.single(xs[self.view - 1])
    #     #fuse = [fused_new, fused_old, single]
    #     #hidden = torch.cat(fuse, dim=1)
    #     #output = self.gate(hidden)
    #         # fused = self.gate([fused_old, single, fused_new])
    #         # output, hidden = self.cluster_module(fused)
    #         # output, hidden = self.cluster_module(fused_old)
    #     return single, single, single, xs[self.view - 1], single
    #     #return zs_old, zs_old, output, hidden



