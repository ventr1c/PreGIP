#%%
import os
import sys
package_dir = '/home/mfl5681/project-ipprotection/pretrain-gnns/chem'
sys.path.append(package_dir)
from util import MaskAtom

import argparse
import numpy as np
import torch
import utils
from torch_geometric.datasets import TUDataset, MoleculeNet, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
import copy

# Dataset settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
<<<<<<< HEAD
parser.add_argument('--num_repeat', type=int, default=50)
=======
parser.add_argument('--num_repeat', type=int, default=5)
>>>>>>> b99538ad237ca2f0286619ea31de9e82c9204f15
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--base_model', type=str, default='GIN', help='propagation model for encoder',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--encoder_model', type=str, default='GraphCL', help='propagation model for encoder',
                    choices=['BGRL-G2L','GraphCL',])
parser.add_argument('--dataset', type=str, default='Tox21', 
                    help='Dataset')
parser.add_argument('--batch_size', default=8192, type=int,
                    help="batch_size of graph dataset")
# GPU setting
parser.add_argument('--device_id', type=int, default=2,
                    help="Threshold of prunning edges")
# Split
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help="Ratio of Training set to train encoder in inductive setting")
# GCL setting
parser.add_argument('--alpha', type=float, default=0.3,
                    help='watermarking')

parser.add_argument('--num_wm', type=int, default=20,
                    help='Number of hidden water marks.')
parser.add_argument('--test_iteration', type=int, default=50)

parser.add_argument('--lr', type=float, default=0.001,
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

parser.add_argument('--pretrain_dataset', type=str, default='ZINC', 
                    help='Dataset')
parser.add_argument('--if_load_watermark', action='store_true', default=False,
                    help='Load watermarked encoder pretrained in ZINC dataset.')
# ZINC masking data setting
parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
parser.add_argument('--mask_edge', type=int, default=1,
                        help='whether to mask edges or not together with atoms')
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


if(args.pretrain_dataset == 'ZINC'):
    from loader import MoleculeDataset
    dataset_root = os.path.join(package_dir,'dataset/')
    dataset_name = 'zinc_standard_agent'
    pretrain_dataset = MoleculeDataset(dataset_root + dataset_name, dataset=dataset_name, transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge))
    '''sample only 200k graphs from ZINC'''
    sampled_idx = utils.obtain_idx_random(args.seed, np.array(list(range(len(pretrain_dataset)))), size = int(len(pretrain_dataset) * 0.1))
    pretrain_dataset = pretrain_dataset[sampled_idx]
else:
    raise NotImplementedError("Not implemented on this dataset")

'''load finetune dataset'''
if(args.dataset in ['PROTEINS', 'FRANKENSTEIN', 'MUTAG', 'ENZYMES', 'NCI1']):
    dataset = TUDataset(root='./data/', name=args.dataset,use_node_attr = True)
elif(args.dataset in ['Tox21', 'ToxCast', 'BBBP','BACE','SIDER','ClinTox','MUV','HIV']):
    from loader import MoleculeDataset
    dataset_root = os.path.join(package_dir,'dataset/')
    dataset_name = args.dataset.lower()
    dataset = MoleculeDataset(dataset_root + dataset_name, dataset=dataset_name)
    
    # dataset = MoleculeNet(root='./data/', name=args.dataset)
    if args.dataset == "Tox21":
        num_tasks = 12
    elif args.dataset == "HIV":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "MUV":
        num_tasks = 17
    elif args.dataset == "BACE":
        num_tasks = 1
    elif args.dataset == "BBBP":
        num_tasks = 1
    elif args.dataset == "ToxCast":
        num_tasks = 617
    elif args.dataset == "SIDER":
        num_tasks = 27
    elif args.dataset == "ClinTox":
        num_tasks = 2

elif(args.dataset == 'ZINC'):
    from loader import MoleculeDataset
    dataset_root = os.path.join(package_dir,'dataset/')
    dataset_name = 'zinc_standard_agent'
    dataset = MoleculeDataset(dataset_root + dataset_name, dataset=dataset_name, transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge))
    '''sample only 200k graphs from ZINC'''
    sampled_idx = utils.obtain_idx_random(args.seed, np.array(list(range(len(dataset)))), size = int(len(dataset) * 0.1))
    dataset = dataset[sampled_idx]
elif(args.dataset == 'MNIST' or args.dataset == 'CIFAR10'):
    dataset = GNNBenchmarkDataset(root='./data/', name=args.dataset)
dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True)

#%%
#%%
'''generate watermark graph'''
from Watermark import WatermarkGraph

key_watermark = []
wm_generator = WatermarkGraph(args, device)
from torch_geometric.data import Data, Batch
idx_pair = utils.obtain_idx_random(args.seed, np.arange(len(pretrain_dataset)), args.num_wm)
for idx in idx_pair:
    key_watermark.append(pretrain_dataset[idx])

key_watermark = Batch.from_data_list(key_watermark)


#%%

from models.DeepSign_Masking import Encoder
from models.Backbones import GIN
import GCL.augmentors as A
import GCL.losses as L
import torch.nn as nn
from GCL.models import DualBranchContrast
from torch.utils.data import Subset
gnn = GIN(args.num_hidden, num_layers=2).to(device)

#%%
aug1 = A.Identity()
aug2 = A.RandomChoice([ A.NodeDropping(pn=0.1),
                        A.FeatureMasking(pf=0.1),
                        A.EdgeRemoving(pe=0.1)], 1)

project = nn.Sequential(
            nn.Linear(args.num_hidden, args.num_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(args.num_hidden, args.num_hidden))
            
gcl = Encoder(args = args, encoder=gnn,project=project,device=device).to(device)
from mask_loader import DataLoaderMasking
pretrain_dataloader = DataLoaderMasking(pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 8)

total_indices = list(range(len(pretrain_dataset)))
# indices_to_keep = [i for i in total_indices if i not in idx_pair]
indices_to_keep = total_indices
# Create a subset with the indices to keep
pretrain_pretrain_dataset = Subset(pretrain_dataset, indices_to_keep)
pretrain_pretrain_loader = DataLoaderMasking(pretrain_pretrain_dataset, batch_size=args.batch_size,shuffle=True, num_workers = 8)
if args.alpha == 0.0:
    if(args.if_load_watermark):
        gcl.encoder.load_state_dict(torch.load("./checkpoint/{}_GCL/deepsign_wm_mask_{}.pt".format('ZINC200k',args.alpha)))
    else:
        gcl.pretrain(pretrain_pretrain_dataset, args,verbose=args.debug)
        torch.save(gcl.encoder.state_dict(), "./checkpoint/{}_GCL/deepsign_wm_mask_{}.pt".format('ZINC200k',args.alpha))
else:   
    if(args.if_load_watermark):
        gcl.encoder.load_state_dict(torch.load("./checkpoint/{}_GCL/deepsign_wm_mask_{}.pt".format('ZINC200k',args.alpha)))
    else:
<<<<<<< HEAD
        gcl.watermarking(key_watermark, pretrain_pretrain_loader, args, verbose=args.debug)
=======
        gcl.watermarking(key_watermark, pretrain_pretrain_dataset, args, verbose=args.debug)
>>>>>>> b99538ad237ca2f0286619ea31de9e82c9204f15
        torch.save(gcl.encoder.state_dict(), "./checkpoint/{}_GCL/deepsign_wm_mask_{}.pt".format('ZINC200k',args.alpha))
#%%
acc_fix_list = []
acc_finetune_list = []

water_IP_fix = []
water_IP_finetune = []


ind_acc_fix_list = []
ind_acc_finetune_list = []

ind_IP_fix = []
ind_IP_finetune = []
for seed in range(10,10+args.num_repeat):
    args.device_id = 3
    device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    split_train = utils.get_split_self(num_samples=len(dataset), train_ratio=0.5, test_ratio=0.4,seed=seed,device=device)
    idx_train = split_train['train']    
    idx_test = split_train['test']
    idx_val = split_train['valid']


    trainloader = DataLoader(Subset(dataset,idx_train), batch_size=args.batch_size)
    testloader = DataLoader(Subset(dataset,idx_test), batch_size=args.batch_size)
    valloader = DataLoader(Subset(dataset,idx_val),batch_size=args.batch_size)

    '''after pretraining, finetune encoder in the '''
    wm_gcl = copy.deepcopy(gcl)
    wm_gcl.transfer_finetune(trainloader, testloader, valloader, args, num_tasks, verbose=args.debug)

    '''Evaluation'''
<<<<<<< HEAD
    acc_fix, IP_fix = wm_gcl.test(trainloader,testloader, key_watermark,args)
    acc_finetune, IP_finetune = wm_gcl.test_finetune(trainloader,testloader, key_watermark,num_tasks,args)
=======
    acc_fix, IP_fix = wm_gcl.test(trainloader,testloader, key_syn, key_watermark,args)
    acc_finetune, IP_finetune = wm_gcl.test_finetune(trainloader,testloader, key_syn, key_watermark,num_tasks,args)
>>>>>>> b99538ad237ca2f0286619ea31de9e82c9204f15

    # print(acc_fix,IP_fix)
    acc_fix_list.append(acc_fix)
    acc_finetune_list.append(acc_finetune)

    water_IP_fix.append(IP_fix)
    water_IP_finetune.append(IP_finetune)

    args.device_id = 2
    device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
    # independ model 
    ind_gnn = GIN(args.num_hidden, num_layers=2).to(device)
    ind_gcl = Encoder(args = args, encoder=ind_gnn,project=project,device=device).to(device)
    ind_gcl.encoder.load_state_dict(torch.load("./checkpoint/{}_Masking/mask_{}.pt".format('ZINC200k', seed)))
    print("Load encoder successfully!")
    '''
    finetune pretrained model
    '''
    ind_gcl.transfer_finetune(trainloader, testloader, valloader, args, num_tasks=num_tasks, verbose=args.debug)
    

<<<<<<< HEAD
    acc_fix, IP_fix = ind_gcl.test(trainloader,testloader, key_watermark,args)
    acc_finetune, IP_finetune = ind_gcl.test_finetune(trainloader,testloader, key_watermark,num_tasks,args)
=======
    acc_fix, IP_fix = ind_gcl.test(trainloader,testloader, key_syn, key_watermark,args)
    acc_finetune, IP_finetune = ind_gcl.test_finetune(trainloader,testloader, key_syn, key_watermark,num_tasks,args)
>>>>>>> b99538ad237ca2f0286619ea31de9e82c9204f15

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

