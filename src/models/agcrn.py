import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveGraphConv(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, cheb_order: int, embed_dim: int):
        super().__init__()
        self.cheb_order = int(cheb_order)
        self.weights = nn.Parameter(torch.empty(embed_dim, cheb_order, dim_in, dim_out))
        self.bias = nn.Parameter(torch.empty(embed_dim, dim_out))
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        nodes = embeddings.size(0)
        support = F.softmax(F.relu(embeddings @ embeddings.transpose(0, 1)), dim=1)
        polynomials = [torch.eye(nodes, device=x.device, dtype=x.dtype)]
        if self.cheb_order > 1:
            polynomials.append(support)
        for _ in range(2, self.cheb_order):
            polynomials.append(2.0 * support @ polynomials[-1] - polynomials[-2])
        supports = torch.stack(polynomials)
        weights = torch.einsum("nd,dkio->nkio", embeddings, self.weights)
        bias = embeddings @ self.bias
        graph_features = torch.einsum("knm,bmc->bknc", supports, x).permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", graph_features, weights) + bias


class AGCRNCell(nn.Module):
    def __init__(self, dim_in: int, hidden_dim: int, cheb_order: int, embed_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gates = AdaptiveGraphConv(dim_in + hidden_dim, 2 * hidden_dim, cheb_order, embed_dim)
        self.candidate = AdaptiveGraphConv(dim_in + hidden_dim, hidden_dim, cheb_order, embed_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        z, r = torch.sigmoid(self.gates(torch.cat([x, state], dim=-1), embeddings)).chunk(2, dim=-1)
        candidate = torch.tanh(self.candidate(torch.cat([x, z * state], dim=-1), embeddings))
        return r * state + (1.0 - r) * candidate


class AGCRN(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, out_dim: int, horizon: int, embed_dim: int = 10,
                 hidden_channels: int = 64, layers: int = 2, cheb_order: int = 2, **_):
        super().__init__()
        self.out_dim, self.horizon, self.hidden_channels = int(out_dim), int(horizon), int(hidden_channels)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        cells = [AGCRNCell(in_dim, hidden_channels, cheb_order, embed_dim)]
        cells.extend(AGCRNCell(hidden_channels, hidden_channels, cheb_order, embed_dim) for _ in range(1, int(layers)))
        self.cells = nn.ModuleList(cells)
        self.output = nn.Linear(hidden_channels, out_dim * horizon)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        batch, _, nodes, steps = x.shape
        current = x.permute(0, 3, 2, 1)
        for cell in self.cells:
            state = torch.zeros(batch, nodes, self.hidden_channels, device=x.device, dtype=x.dtype)
            outputs = []
            for step in range(steps):
                state = cell(current[:, step], state, self.node_embeddings)
                outputs.append(state)
            current = torch.stack(outputs, dim=1)
        output = self.output(current[:, -1])
        return output.view(batch, nodes, self.out_dim, self.horizon).permute(0, 2, 1, 3)
