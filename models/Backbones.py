
import torch
from torch import nn
from torch_geometric.nn import GINEConv,global_add_pool, global_mean_pool
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

def make_gin_conv(input_dim, output_dim):
    return GINEConv(nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Linear(output_dim,output_dim)))


class GIN(nn.Module):
    '''
    Only used the final layer presentation instead of the representation of multiple layers
    Not sure whether we should use batch normalization
    '''
    def __init__(self, hidden_dim, num_layers, dropout=0.0):
        super(GIN, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, hidden_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, hidden_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, hidden_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, hidden_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr, batch, edge_weight=None):
        # update edge_index if edge_weight is not None
        if(edge_weight is not None):
            edge_index = edge_index[:, edge_weight > 0]
        # print(x, edge_attr)
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        z = x
        # print(z, edge_embeddings)
        for conv in self.layers:
            z = conv(z, edge_index, edge_embeddings)
            z = F.relu(z)
            # z = bn(z)
        g = global_mean_pool(z, batch)
        g = F.dropout(g, self.dropout, training=self.training)
        return z, g
    

