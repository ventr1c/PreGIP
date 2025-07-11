#%%
import os
import time
import argparse
import numpy as np
import torch
import utils
import copy
import yaml
from yaml import SafeLoader


from torch_geometric.datasets import TUDataset, MoleculeNet, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

# Dataset settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--num_repeat', type=int, default=1)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--base_model', type=str, default='GCN', help='propagation model for encoder',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--encoder_model', type=str, default='GraphCL', help='propagation model for encoder',
                    choices=['BGRL-G2L','GraphCL',])
parser.add_argument('--dataset', type=str, default='PROTEINS', 
                    help='Dataset')
parser.add_argument('--batch_size', default=128, type=int,
                    help="batch_size of graph dataset")
# GPU setting
parser.add_argument('--device_id', type=int, default=2,
                    help="Threshold of prunning edges")
# Split
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help="Ratio of Training set to train encoder in inductive setting")
# GCL setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--num_hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--num_proj_hidden', type=int, default=32,
                    help='Number of hidden units in MLP.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--encoder_layer', type=int, default=2, help='layer number of encder')
parser.add_argument('--drop_edge_rate_1', type=float, default=0.1)
parser.add_argument('--drop_edge_rate_2', type=float, default=0.1)
parser.add_argument('--drop_feat_rate_1', type=float, default=0.1)
parser.add_argument('--drop_feat_rate_2', type=float, default=0.1)
parser.add_argument('--drop_node_rate_1', type=float, default=0.1)
parser.add_argument('--drop_node_rate_2', type=float, default=0.1)
parser.add_argument('--walk_length_2', default=10, type=int,
                    help="length of random walk")
parser.add_argument('--tau', type=float, default=0.2)
parser.add_argument('--num_epochs', type=int, default=100)


args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])


if(args.dataset == 'PROTEINS' or args.dataset == 'MUTAG' or args.dataset == 'COLLAB' or args.dataset == 'ENZYMES'):
    dataset = TUDataset(root='./data/', name=args.dataset, transform=transform,use_node_attr = True)
elif(args.dataset == 'Tox21' or args.dataset == 'ToxCast' or args.dataset == 'BBBP'):
    dataset = MoleculeNet(root='./data/', name=args.dataset, transform=None)
elif(args.dataset == 'MNIST' or args.dataset == 'CIFAR10'):
    dataset = GNNBenchmarkDataset(root='./data/', name=args.dataset, transform=transform)


from torch_geometric.utils import to_undirected
for data in dataset:
    data.edge_index = to_undirected(data.edge_index)



config_path = "./config/config_{}.yaml".format(args.encoder_model)   
config = yaml.load(open(config_path), Loader=SafeLoader)[args.dataset]

args.drop_edge_rate_1 = config['drop_edge_rate_1']
args.drop_edge_rate_2 = config['drop_edge_rate_2']
args.drop_feat_rate_1 = config['drop_feat_rate_1']
args.drop_feat_rate_2 = config['drop_feat_rate_2']
if(args.encoder_model == 'GraphCL'):
    args.drop_node_rate_1 = config['drop_node_rate_1']
    args.drop_node_rate_2 = config['drop_node_rate_2']
args.tau = config['tau']
args.lr = config['cl_lr']
args.weight_decay = config['weight_decay']
args.num_epochs = config['cl_num_epochs']
args.num_hidden = config['num_hidden']
args.num_proj_hidden = config['num_proj_hidden']
print(args)

'''generate watermark graph'''
from Watermark import WatermarkGraph
poison_ratio = 0.95
poison_num = int(len(dataset) * poison_ratio)
wm_generator = WatermarkGraph(args, device)

# for i in range(len(dataset)):
#     dataset[i].idx_pair = -1
#     print(dataset[i])
#     break
# for i in range(len(dataset)):
#     dataset[i] = dataset[i].to(device)
idx_pair = utils.obtain_idx_random(args, np.arange(len(dataset)), poison_num)

watermark_dataset = []
for idx in idx_pair:
    data = dataset[idx]
    watermark_data = copy.deepcopy(data)
    num_nodes = data.x.shape[0]
    num_nodes = max(int(num_nodes * 0.15),3)
    p = 0.5
    watermark_feat, watermark_edge_index, watermark_edge_weight = wm_generator.gene_random_graph(data.x, data.edge_index, data.edge_weight, num_nodes, p = p, feat_generation = 'Rand_Gene')
    watermark_data.x = watermark_feat
    watermark_data.edge_index = watermark_edge_index
    watermark_data.edge_weight = watermark_edge_weight
    watermark_dataset.append(watermark_data)
    watermark_data.idx_pair = idx
# concat watermark dataset to original dataset
concat_dataset = dataset + watermark_dataset
concat_dataset.num_features = dataset.num_features

print(len(watermark_dataset), len(dataset[idx_pair]))
'''
Train GNN encoder via GCL. For watermarked graph, we add a 
regularization term to the loss function to make the representations 
of watermarked graph and original graph similar. 
'''
from models.construct import model_construct_global

rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=args.num_repeat)

split_train = utils.get_split_self(num_samples=len(dataset), train_ratio=0.1, test_ratio=0.8,device=device)
idx_train = split_train['train']    
idx_val = split_train['valid']

idx_train = np.setdiff1d(idx_train.cpu(), idx_pair)
idx_val = np.setdiff1d(idx_val.cpu(), idx_pair)

idx_train = torch.tensor(idx_train).to(device)
idx_val = torch.tensor(idx_val).to(device)

idx_watermark = np.arange(len(dataset),len(dataset)+len(watermark_dataset))
idx_watermark = torch.tensor(idx_watermark).to(device)

for seed in seeds:
    np.random.seed(seed)
    print("seed {}".format(seed))
    model = model_construct_global(args,args.encoder_model, concat_dataset, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    # train the encoder on the original dataset
    model.fit(dataloader, None, train_iters=args.num_epochs,watermark=False,verbose=True)
    concat_dataloader = DataLoader(concat_dataset, batch_size=args.batch_size)
    pair_result = model.test(dataloader, idx_train, idx_pair, idx_val)
    wm_result = model.test(concat_dataloader, idx_train, idx_watermark, idx_val)
    print("before watermarking")
    print(pair_result, wm_result)
    # train the encoder on the watermarked dataset
    watermark_dataloader = DataLoader(watermark_dataset, batch_size=args.batch_size)
    pair_dataloader = DataLoader(dataset[idx_pair], batch_size=args.batch_size)
    model.fit(watermark_dataloader, pair_dataloader, train_iters=args.num_epochs*2,watermark=True,verbose=True)
    # evaluation on downstream tasks
    concat_dataloader = DataLoader(concat_dataset, batch_size=args.batch_size)
    pair_result = model.test(concat_dataloader, idx_train, idx_pair, idx_val)
    wm_result = model.test(concat_dataloader, idx_train, idx_watermark, idx_val)
    print("after watermarking")
    print(pair_result, wm_result)