import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv,GATConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

from copy import deepcopy
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.utils import subgraph
from torch_geometric.utils import to_undirected, to_dense_adj,to_torch_coo_tensor,dense_to_sparse,degree
import torch_geometric.utils as pyg_utils
import eval

# from models.MLP import MLP

class GCN_body(nn.Module):
    def __init__(self,nfeat, nhid, dropout=0.5, layer=2,device=None,layer_norm_first=False,use_ln=False):
        super(GCN_body, self).__init__()
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat,nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(1, layer-1):
            self.convs.append(GCNConv(nhid,nhid))
            self.lns.append(nn.LayerNorm(nhid))
            
        self.convs.append(GCNConv(nhid,nhid))
        self.lns.append(nn.LayerNorm(nhid))

        # self.convs.append(GCNConv(nfeat,2 * nhid))
        # self.lns = nn.ModuleList()
        # self.lns.append(torch.nn.LayerNorm(nfeat))
        # for _ in range(1, layer-1):
        #     self.convs.append(GCNConv(2 * nhid,2 * nhid))
        #     self.lns.append(nn.LayerNorm(2 * nhid))
            
        # self.convs.append(GCNConv(2 * nhid,nhid))
        # self.lns.append(nn.LayerNorm(2 * nhid))

        self.lns.append(torch.nn.LayerNorm(nhid))
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
    def forward(self,x, edge_index,edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln: 
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        return x

    def inference(self, x_all, subgraph_loader):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)
                x = x_all[n_id].to(self.device)
                x_target = x[:size[1]]
                x = conv(x, edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

class GAT_body(nn.Module):
    def __init__(self,nfeat, nhid, heads=8, dropout=0.5, layer=2,device=None,layer_norm_first=False,use_ln=False):
        super(GAT_body, self).__init__()
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(nfeat,nhid,heads,dropout=dropout))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(1, layer-1):
            self.convs.append(GATConv(heads*nhid,heads*nhid,dropout=dropout))
            self.lns.append(nn.LayerNorm(heads*nhid))
            
        self.convs.append(GATConv(heads*nhid,nhid,concat=False, dropout=dropout))
        self.lns.append(nn.LayerNorm(nhid))

        # self.convs.append(GCNConv(nfeat,2 * nhid))
        # self.lns = nn.ModuleList()
        # self.lns.append(torch.nn.LayerNorm(nfeat))
        # for _ in range(1, layer-1):
        #     self.convs.append(GCNConv(2 * nhid,2 * nhid))
        #     self.lns.append(nn.LayerNorm(2 * nhid))
            
        # self.convs.append(GCNConv(2 * nhid,nhid))
        # self.lns.append(nn.LayerNorm(2 * nhid))

        self.lns.append(torch.nn.LayerNorm(nhid))
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
    def forward(self,x, edge_index,edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln: 
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        return x

class Grace(nn.Module):

    def __init__(self, args, nfeat, nhid, nproj, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,device=None,use_ln=False,layer_norm_first=False):

        super(Grace, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.use_ln = use_ln

        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay
        self.tau = tau
        self.args = args
        self.layer_norm_first = layer_norm_first
        # self.convs = nn.ModuleList()
        # self.convs.append(GCNConv(nfeat, nhid))
        # for _ in range(layer-2):
        #     self.convs.append(GCNConv(nhid,nhid))
        # self.gc2 = GCNConv(nhid, nclass)
        if(args.base_model == 'GCN'):
            self.body = GCN_body(nfeat, nhid, dropout, layer,device=device,use_ln=use_ln,layer_norm_first=layer_norm_first).to(device)
        elif(args.base_model=='GAT'):
            self.body = GAT_body(nfeat, nhid, 8, dropout, layer,device=device,use_ln=use_ln,layer_norm_first=layer_norm_first).to(device)
        # self.body = MLP(nfeat,nhid,nhid,dropout,lr=lr, weight_decay=weight_decay, device=device)
        # linear evaluation layer
        self.fc = nn.Linear(nhid,nclass).to(device)

        # projection layer
        self.fc1 = torch.nn.Linear(nhid, nproj).to(device)
        self.fc2 = torch.nn.Linear(nproj, nhid).to(device)

    
    def forward(self, x, edge_index, edge_weight=None):
        # for conv in self.convs:
        #     x = F.relu(conv(x, edge_index,edge_weight))
        #     x = F.dropout(x, self.dropout, training=self.training)
        x = self.body(x, edge_index,edge_weight)
        # x = self.fc(x)
        # return F.log_softmax(x,dim=1)
        return x
    def get_h(self, x, edge_index,edge_weight=None):
        self.eval()
        x = self.body(x, edge_index,edge_weight)
        # for conv in self.convs:
        #     x = F.relu(conv(x, edge_index))
        return x
    
    def aug_forward(self, x, edge_index, edge_weight=None):
        edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_overall(self.args, x, edge_index, edge_weight, device= self.device)
        z = self.body(x_1, edge_index_1,edge_weight_1)
        return z
    
    def aug_forward1(self, x, edge_index, edge_weight=None):
        edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_overall(self.args, x, edge_index, edge_weight, device= self.device)
        # idx_1 = torch.randperm(x.shape[0])
        # x_1 = x[idx_1,:]
        # idx_2 = torch.randperm(x.shape[0])
        # x_2 = x[idx_2,:]
        z1 = self.body(x_1, edge_index_1,edge_weight_1)
        z2 = self.body(x_2, edge_index_2,edge_weight_2)
        return z1,z2

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
  
    def fit(self, features, edge_index, edge_weight, labels, train_iters=200,seen_node_idx=None,idx_train=None,idx_val=None, idx_test=None, verbose=False):
        # indices_target_seen, indices_target_test
        # self.indices_target_seen = indices_target_seen
        # self.indices_target_test = indices_target_test
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.seen_node_idx = seen_node_idx
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        # self._train_with_val(self.labels, train_iters, verbose)
        self._train_without_val(self.labels, train_iters, verbose)
        # self._train_without_val_batch(self.labels, train_iters, verbose)
        # if idx_val is None:
        #     self._train_without_val(self.labels, idx_train, train_iters, verbose)
        # else:
        #     self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        train_list = []
        test_list = []
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            edge_index, edge_weight = self.edge_index, self.edge_weight
            edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_overall(self.args, self.features, edge_index, edge_weight, device= self.device)

            z1 = self.forward(x_1, edge_index_1,edge_weight_1)
            z2 = self.forward(x_2, edge_index_2,edge_weight_2)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2
            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=self.args.cont_batch_size)
            else:
                cont_loss = self.loss(h1, h2, batch_size=self.args.cont_batch_size)

            loss =  cont_loss
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss.item()))
                target_train_view_sims = F.cosine_similarity(h1[self.indices_target_seen],h2[self.indices_target_seen]).sum()
                target_test_view_sims = F.cosine_similarity(h1[self.indices_target_test],h2[self.indices_target_test]).sum()
                train_list.append(target_train_view_sims.item())
                test_list.append(target_test_view_sims.item())
                print(target_train_view_sims,target_test_view_sims)
            loss.backward()
            optimizer.step()
        print(train_list)
        print(test_list)

    def _train_with_val(self, labels, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            # edge_index, edge_weight = self.sample_noise_all(self.edge_index,self.edge_weight,idx_train)
            edge_index, edge_weight = self.edge_index, self.edge_weight
            # edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_1by1(self.args, self.features, edge_index, edge_weight)
            edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_overall(self.args, self.features, edge_index, edge_weight, device= self.device)
    
            z1 = self.forward(x_1, edge_index_1,edge_weight_1)
            z2 = self.forward(x_2, edge_index_2,edge_weight_2)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2

            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=self.args.cont_batch_size)
            else:
                cont_loss = self.loss(h1, h2, batch_size=self.args.cont_batch_size)

            loss =  cont_loss
            loss.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss.item()))
            if i % 50 == 0:
                self.eval()
                z = self.forward(self.features, self.edge_index,self.edge_weight)
                acc_val = eval.lr_evaluation(z,labels,self.idx_train,self.idx_val)
                if acc_val>best_acc_val:
                    best_acc_val = acc_val
                    self.weights = deepcopy(self.state_dict())
                print('Epoch {}, training loss: {} val acc: {} best val acc: {}'.format(i, loss.item(), acc_val, best_acc_val))
        self.load_state_dict(self.weights)

    def test(self, features, edge_index, edge_weight, labels, idx_train, idx_test, idx_val=None):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        idx_train = torch.tensor(idx_train)
        idx_test = torch.tensor(idx_test)
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc, pred = eval.linear_evaluation(output, labels, idx_train, idx_test)
        
        results = {}
        results['accuracy'] = acc
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return acc_test,correct_nids
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1[mask]))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2[mask]))  # [B, N]

            losses.append(-torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())))

            # losses.append(-torch.log(
            #     between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            #     / (refl_sim.sum(1) + between_sim.sum(1)
            #        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        # h1 = z1
        # h2 = z2

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def clf_loss(self, z: torch.Tensor, labels, idx):
        # h = self.projection(z)
        h = z
        # print(z,labels,idx)
        output = self.clf_head(h)
        
        clf_loss = F.nll_loss(output[idx],labels[idx])
        return clf_loss
    
    def clf_head(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc(x)
        return F.log_softmax(z,dim=1)
        # z = F.elu(self.fc1_c(z))
        # return self.fc2_c(z)
        # return z
  
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def drop_adj_1by1(args,edge_index, edge_weight, p,device):
    # update edge_index according to edge_weight
    if(edge_weight!=None):
        edge_index = edge_index[:,edge_weight.nonzero().flatten().long()]
        edge_weight = torch.ones([edge_index.shape[1],]).to(device)

    remain_mask = np.random.binomial(1,p,edge_index.shape[1])
    remain_index = remain_mask.nonzero()[0]
    remain_edge_index = edge_index[:,remain_index]
    remain_edge_weight = torch.ones([remain_edge_index.shape[1],]).to(device)
    return remain_edge_index,remain_edge_weight
  
def construct_augmentation_overall(args, x, edge_index, edge_weight=None, device=None):
    # graph 1:
    aug_edge_index_1,aug_edge_weight_1 = drop_adj_1by1(args,edge_index, edge_weight, 1-args.drop_edge_rate_1,device)
    aug_edge_index_1 = aug_edge_index_1.long()

    aug_x_1 = drop_feature(x,drop_prob=args.drop_feat_rate_1)

    # graph 2:
    aug_edge_index_2,aug_edge_weight_2 = drop_adj_1by1(args,edge_index, edge_weight, 1-args.drop_edge_rate_2,device)
    aug_edge_index_2 = aug_edge_index_2.long()

    aug_x_2 = drop_feature(x,drop_prob=args.drop_feat_rate_2)
    return aug_edge_index_1,aug_x_1,aug_edge_weight_1,aug_edge_index_2,aug_x_2,aug_edge_weight_2