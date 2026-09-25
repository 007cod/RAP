import torch
from torch import nn
import torch.nn.functional as F


class NConv(nn.Module):
    def forward(self, x, adj):
        return torch.einsum("bcnt,nm->bcmt", (x, adj)).contiguous()


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super().__init__()
        self.nconv = NConv()
        self.mlp = nn.Conv2d((order * support_len + 1) * c_in, c_out, kernel_size=(1, 1))
        self.dropout = dropout
        self.order = order

    def forward(self, x, supports):
        out = [x]
        for adj in supports:
            x1 = self.nconv(x, adj)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, adj)
                out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_dim,
        out_dim,
        horizon,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=64,
        end_channels=128,
        blocks=2,
        layers=3,
        dropout=0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = 1
            for layer in range(layers):
                dilation = 2**layer
                self.filter_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, 2), dilation=(1, dilation))
                )
                self.gate_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, 2), dilation=(1, dilation))
                )
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.gconv.append(GraphConv(dilation_channels, residual_channels, dropout, support_len=2, order=2))
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim * horizon, kernel_size=(1, 1), bias=True)

    def forward(self, x, adj):
        # x: [B, C, N, T]
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))
        supports = [adj, F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)]
        x = self.start_conv(x)
        skip = None
        for i in range(self.blocks * self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate
            s = self.skip_convs[i](x)
            skip = s if skip is None else s + skip[:, :, :, -s.size(3) :]
            x = self.gconv[i](x, supports)
            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x[:, :, :, -1:]
        b, c, n, _ = x.shape
        return x.view(b, -1, self.horizon, n).permute(0, 1, 3, 2).contiguous()


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    adj = adj + torch.eye(adj.shape[0], device=adj.device)
    degree = adj.sum(dim=1).clamp_min(1e-6)
    return adj / degree[:, None]
