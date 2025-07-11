
import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
import torch.nn.functional as F

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GIN_graphCL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GIN_graphCL, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g
    


class GIN(nn.Module):
    '''
    Only used the final layer presentation instead of the representation of multiple layers
    Not sure whether we should use batch normalization
    '''
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super(GIN, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index)
            z = F.relu(z)
            # z = bn(z)
        g = global_mean_pool(z, batch)
        g = F.dropout(g, self.dropout, training=self.training)
        return z, g
    

