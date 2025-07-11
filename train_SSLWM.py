#%%
import argparse
import numpy as np
import torch
import utils
from torch_geometric.datasets import TUDataset, MoleculeNet, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

# Dataset settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--num_repeat', type=int, default=50)
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
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
# Split
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help="Ratio of Training set to train encoder in inductive setting")
# GCL setting
parser.add_argument('--alpha', type=float, default=0.3,
                    help='watermarking')

parser.add_argument('--num_wm', type=int, default=40,
                    help='Number of hidden water marks.')
parser.add_argument('--watermark_size', type=int, default=5,
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
parser.add_argument('--num_epochs', type=int, default=200)


args = parser.parse_known_args()[0]

print(args)

args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

import random
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#%%

if(args.dataset == 'PROTEINS' or args.dataset == 'MUTAG' or args.dataset == 'COLLAB' or args.dataset == 'ENZYMES'):
    dataset = TUDataset(root='./data/', name=args.dataset,use_node_attr = True)
elif(args.dataset == 'Tox21' or args.dataset == 'ToxCast' or args.dataset == 'BBBP'):
    dataset = MoleculeNet(root='./data/', name=args.dataset)
elif(args.dataset == 'MNIST' or args.dataset == 'CIFAR10'):
    dataset = GNNBenchmarkDataset(root='./data/', name=args.dataset)
#%%
#%%
'''generate watermark graph'''
from Watermark import WatermarkGraph
from utils import inject_trigger
from torch_geometric.data import Data, Batch
wm_generator = WatermarkGraph(args, device)
wm_graph = wm_generator.gene_random_graph(dataset.data.x,  args.watermark_size, p = 0.8, feat_generation = 'Rand_Gene')
wm_graph = Data(x=wm_graph[0],edge_index=wm_graph[1])
key_watermark = []
idx_pair = utils.obtain_idx_random(args.seed, np.arange(len(dataset)), args.num_wm)
for idx in idx_pair:
    key_watermark.append(inject_trigger(0,dataset[idx],wm_graph))

key_watermark = Batch.from_data_list(key_watermark)


#%%

from models.SSLWM import Encoder
from models.Backbones import GIN
import GCL.augmentors as A
import GCL.losses as L
import torch.nn as nn
from GCL.models import DualBranchContrast
from torch.utils.data import Subset
gnn = GIN(dataset.num_features, args.num_hidden, num_layers=2).to(device)

#%%
aug1 = A.Identity()
aug2 = A.RandomChoice([ A.NodeDropping(pn=0.1),
                        A.FeatureMasking(pf=0.1),
                        A.EdgeRemoving(pe=0.1)], 1)

project = nn.Sequential(
            nn.Linear(args.num_hidden, args.num_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(args.num_hidden, args.num_hidden))

contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
gcl = Encoder(encoder=gnn,augmentor=(aug1,aug2),project=project, contrast_model=contrast_model,watermark=wm_graph,device=device).to(device)
total_indices = list(range(len(dataset)))
# indices_to_keep = [i for i in total_indices if i not in idx_pair]
indices_to_keep = total_indices
# Create a subset with the indices to keep
pretrain_dataset = Subset(dataset, indices_to_keep)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size,shuffle=True)
if args.alpha == 0.0:
    gcl.pretrain(pretrain_loader, args,verbose=args.debug)
else:   
    gcl.watermarking(key_watermark, pretrain_loader, args, verbose=args.debug)

#%%
acc_fix_list = []
acc_finetune_list = []

water_IP_fix = []
water_IP_finetune = []


ind_acc_fix_list = []
ind_acc_finetune_list = []

ind_IP_fix = []
ind_IP_finetune = []

test_watermark = []
idx_pair = utils.obtain_idx_random(args.seed+1, np.arange(len(dataset)), args.num_wm)
for idx in idx_pair:
    test_watermark.append(inject_trigger(0,dataset[idx],wm_graph))
test_watermark = Batch.from_data_list(test_watermark)

for seed in range(10,10+args.num_repeat):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    split_train = utils.get_split_self(num_samples=len(dataset), train_ratio=0.5, test_ratio=0.4,seed=seed,device=device)
    idx_train = split_train['train']    
    idx_test = split_train['test']

    trainloader = DataLoader(Subset(dataset,idx_train), batch_size=args.batch_size)
    testloader = DataLoader(Subset(dataset,idx_test), batch_size=args.batch_size)

    acc_fix, IP_fix = gcl.test(trainloader,testloader, test_watermark,args)
    acc_finetune, IP_finetune = gcl.test_finetune(trainloader,testloader, test_watermark,args)

    acc_fix_list.append(acc_fix)
    acc_finetune_list.append(acc_finetune)

    water_IP_fix.append(IP_fix)
    water_IP_finetune.append(IP_finetune)

    # independ model 
    ind_gnn = GIN(dataset.num_features, args.num_hidden, num_layers=2).to(device)
    ind_gcl = Encoder(encoder=ind_gnn,augmentor=(aug1,aug2),project=project, contrast_model=contrast_model,watermark=wm_graph,device=device).to(device)
    ind_gcl.encoder.load_state_dict(torch.load("./checkpoint/GCL/gcl_{}.pt".format(seed)))

    acc_fix, IP_fix = ind_gcl.test(trainloader,testloader, test_watermark,args)
    acc_finetune, IP_finetune = ind_gcl.test_finetune(trainloader,testloader, test_watermark,args)

    ind_acc_fix_list.append(acc_fix)
    ind_acc_finetune_list.append(acc_finetune)
    ind_IP_fix.append(IP_fix)
    ind_IP_finetune.append(IP_finetune)


print("=============IP Score=============")
print("IP Score (Fixed), Indepdenpend: {:.4f}, {:.4f}, Watermarking  {:.4f}, {:.4f}"\
      .format(np.mean(ind_IP_fix), np.std(ind_IP_fix), np.mean(water_IP_fix),np.std(water_IP_fix)))

print("IP Score (Fine-tune), Indepdenpend: {:.4f}, {:.4f}, Watermarking  {:.4f}, {:.4f}"\
      .format(np.mean(ind_IP_finetune), np.std(ind_IP_finetune), np.mean(water_IP_finetune),np.std(water_IP_finetune)))

print("=============Accuracy=============")
print("Accuracy (Fixed), Indepdenpend: {:.4f}, {:.4f}, Watermarking  {:.4f}, {:.4f}"\
      .format(np.mean(ind_acc_fix_list), np.std(ind_acc_fix_list), np.mean(acc_fix_list),np.std(acc_fix_list)))
print("Accuracy (Fine-tune), Indepdenpend: {:.4f}, {:.4f}, Watermarking  {:.4f}, {:.4f}"\
      .format(np.mean(ind_acc_finetune_list), np.std(ind_acc_finetune_list), np.mean(acc_finetune_list),np.std(acc_finetune_list)))


print("=============IP Evaluation=============")
# %%
# %%
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score

y = np.concatenate([np.zeros(args.num_repeat), np.ones(args.num_repeat)]) 

score_fix = roc_auc_score(y, np.concatenate([ind_IP_fix,water_IP_fix]))
diff_fix = np.asarray(water_IP_fix) - np.asarray(ind_IP_fix)
print("ROC score (Fixed): {:.4f}, p-value: {}, Difference: {:.4f}, {:.4f}"\
      .format(score_fix,ttest_rel(ind_IP_fix, water_IP_fix).pvalue, np.mean(diff_fix), np.std(diff_fix)))


diff_finetune = np.asarray(water_IP_finetune) - np.asarray(ind_IP_finetune)
score_finetune = roc_auc_score(y, np.concatenate([ind_IP_finetune,water_IP_finetune]))
print("ROC score (Finetune): {:.4f}, p-value: {}, Difference: {:.4f}, {:.4f}"\
      .format(score_finetune,ttest_rel(ind_IP_finetune, water_IP_finetune).pvalue, np.mean(diff_finetune), np.std(diff_finetune)))


# %%
