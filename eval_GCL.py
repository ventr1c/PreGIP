#%%
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
        default=False, help='debug mode')
parser.add_argument('--num_repeat', type=int, default=1)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--base_model', type=str, default='GIN', help='propagation model for encoder',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--encoder_model', type=str, default='GraphCL', help='propagation model for encoder',
                    choices=['BGRL-G2L','GraphCL',])
parser.add_argument('--dataset', type=str, default='PROTEINS', 
                    help='Dataset')
parser.add_argument('--batch_size', default=256, type=int,
                    help="batch_size of graph dataset")
# GPU setting
parser.add_argument('--device_id', type=int, default=3,
                    help="Threshold of prunning edges")
# Split
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help="Ratio of Training set to train encoder in inductive setting")
parser.add_argument('--eps', type=float, default=2.0,
                    help='size of the ball')
parser.add_argument('--step', type=int, default=10,
                    help='number of steps in meta IP')
parser.add_argument('--second_order', action='store_true',
        default=False, help='whether compute second order gradient')
parser.add_argument('--random', action='store_true',
        default=False, help='whether compute second order gradient')

parser.add_argument('--num_wm', type=int, default=20,
                    help='Number of hidden water marks.')
parser.add_argument('--watermark_size', type=int, default=1,
                    help='Number of hidden water marks.')
parser.add_argument('--test_iteration', type=int, default=50)

parser.add_argument('--lr', type=float, default=0.0003,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--num_hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--num_proj_hidden', type=int, default=128,
                    help='Number of hidden units in MLP.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--num_epochs', type=int, default=500)


args = parser.parse_known_args()[0]

print(args)

args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))


#%%

if(args.dataset == 'PROTEINS' or args.dataset == 'MUTAG' or args.dataset == 'COLLAB' or args.dataset == 'ENZYMES'):
    dataset = TUDataset(root='./data/', name=args.dataset,use_node_attr = True)
elif(args.dataset == 'Tox21' or args.dataset == 'ToxCast' or args.dataset == 'BBBP'):
    dataset = MoleculeNet(root='./data/', name=args.dataset)
elif(args.dataset == 'MNIST' or args.dataset == 'CIFAR10'):
    dataset = GNNBenchmarkDataset(root='./data/', name=args.dataset)

#%%

sizes = []
for g in dataset:
    sizes.append(g.x.shape[0])

#%%
import random
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


'''generate watermark graph'''
from Watermark import WatermarkGraph
key_watermark = []
key_syn = []
key_all = []
wm_generator = WatermarkGraph(args, device)
from torch_geometric.data import Data, Batch
for _ in range(args.num_wm):
    # key_normal.append(dataset[idx])

    p = 0.2
    num_nodes = np.clip(int(np.random.normal(np.mean(sizes),np.std(sizes))),1,100)
    watermark_feat, watermark_edge_index, watermark_edge_weight = \
                    wm_generator.gene_random_graph(dataset.data.x, num_nodes, p = p, feat_generation = 'Rand_Gene')
    key_syn.append(Data(x=watermark_feat,edge_index=watermark_edge_index))
    key_all.append(Data(x=watermark_feat,edge_index=watermark_edge_index))
    num_nodes = np.clip(int(np.random.normal(np.mean(sizes),np.std(sizes))),1,100)
    watermark_feat, watermark_edge_index, watermark_edge_weight = \
                    wm_generator.gene_random_graph(dataset.data.x, num_nodes, p = 0.2, feat_generation = 'Rand_Gene')
    key_watermark.append(Data(x=watermark_feat,edge_index=watermark_edge_index))
    key_all.append(Data(x=watermark_feat,edge_index=watermark_edge_index))
key_syn = Batch.from_data_list(key_syn)
key_watermark = Batch.from_data_list(key_watermark)

key_all = Batch.from_data_list(key_all)

#%%

from models.IPGCL import Encoder
from models.Backbones import GIN
import GCL.augmentors as A
import GCL.losses as L
import torch.nn as nn
from GCL.models import DualBranchContrast

acc_fix_list = []
acc_finetune_list = []

IP_fix_list = []
IP_finetune_list = []
for seed in range(10,20):


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    gnn = GIN(dataset.num_features, args.num_hidden, num_layers=2).to(device)
    aug1 = A.Identity()
    aug2 = A.RandomChoice([ A.NodeDropping(pn=0.1),
                            A.FeatureMasking(pf=0.1),
                            A.EdgeRemoving(pe=0.1)], 1)

    project = nn.Sequential(
                nn.Linear(args.num_hidden, args.num_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(args.num_hidden, args.num_hidden))

    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    gcl = Encoder(encoder=gnn,augmentor=(aug1,aug2),project=project, contrast_model=contrast_model,device=device).to(device)
    

    dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True)
    # gcl.pretrain(dataloader, args,verbose=args.debug)
    # torch.save(gcl.encoder.state_dict(), "./checkpoint/GCL/gcl_{}.pt".format(seed))

    gcl.encoder.load_state_dict(torch.load("./checkpoint/GCL/gcl_{}.pt".format(seed)))

    split_train = utils.get_split_self(num_samples=len(dataset), train_ratio=0.5, test_ratio=0.4,seed=seed,device=device)
    idx_train = split_train['train']    
    idx_test = split_train['test']
    from torch.utils.data import Subset
    trainloader = DataLoader(Subset(dataset,idx_train), batch_size=args.batch_size,shuffle=True)
    testloader = DataLoader(Subset(dataset,idx_test), batch_size=args.batch_size)
    acc_fix, IP = gcl.test(trainloader,testloader,key_syn, key_watermark,args)
    acc_fix_list.append(acc_fix)

    print("Fixed:{}".format(IP))

    IP_fix_list.append(IP)

    acc_finetune, IP = gcl.test_finetune(trainloader,testloader,key_syn, key_watermark,args)
    acc_finetune_list.append(acc_finetune)
    IP_finetune_list.append(IP)

    print("Finetune: {}".format(IP))

print("Fix IP: {:.4f}, {:.4f}".format(np.mean(IP_fix_list),np.std(IP_fix_list)))
print("Finetune IP: {:.4f}, {:.4f}".format(np.mean(IP_finetune_list),np.std(IP_finetune_list)))


# %%
