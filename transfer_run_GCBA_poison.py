#%%
import os
import sys
package_dir = '/home/mfl5681/project-ipprotection/pretrain-gnns/chem'
sys.path.append(package_dir)

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
parser.add_argument('--num_repeat', type=int, default=5)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=127, help='Random seed.')
parser.add_argument('--base_model', type=str, default='GIN', help='propagation model for encoder',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--encoder_model', type=str, default='GraphCL', help='propagation model for encoder',
                    choices=['BGRL-G2L','GraphCL',])
parser.add_argument('--dataset', type=str, default='Tox21', 
                    help='Dataset')
parser.add_argument('--pretrain_dataset', type=str, default='ZINC', 
                    help='Dataset')
parser.add_argument('--batch_size', default=256, type=int,
                    help="batch_size of graph dataset")
# GPU setting
parser.add_argument('--device_id', type=int, default=3,
                    help="Threshold of prunning edges")
# Split
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help="Ratio of Training set to train encoder in inductive setting")
# GCL setting
parser.add_argument('--alpha', type=float, default=1.0,
                    help='watermarking')

parser.add_argument('--eps', type=float, default=2.0,
                    help='size of the ball')
parser.add_argument('--step', type=int, default=10,
                    help='number of steps in meta IP')
parser.add_argument('--step_test', type=int, default=30,
                    help='number of steps in fune-tune testing')
parser.add_argument('--second_order', action='store_true',
        default=False, help='whether compute second order gradient')
parser.add_argument('--random', action='store_true',
        default=False, help='whether compute second order gradient')

parser.add_argument('--num_wm', type=int, default=20,
                    help='Number of hidden water marks.')
parser.add_argument('--watermark_size', type=int, default=1,
                    help='Number of hidden water marks.')

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
parser.add_argument('--num_epochs_wm', type=int, default=500)
parser.add_argument('--normal_selection', type=str, default='cluster', choices=['random', 'distance', 'cluster'])

# GCBA setting
parser.add_argument('--bkd_number', type=int, default=20,
                    help='number of backdoored graphs')
parser.add_argument('--clr_number', type=int, default=20,
                    help='number of clean graphs')
parser.add_argument('--ref_number', type=int, default=20,
                    help='number of reference graphs')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='size of trigger')
parser.add_argument('--target_class', type=int, default=1,
                    help='Target class of backdoored graphs')
parser.add_argument('--feat_contineous', action='store_true',
        default=True, help='whether use continuous features for triggers')
parser.add_argument('--trojan_epochs', type=int, default=50,
                    help='number of epochs to train trojan')
parser.add_argument('--inner', type=int, default=1,
                    help='number of inner loop')
parser.add_argument('--trojan_lr', type=float, default=0.01,
                    help='learning rate of trojan network')
parser.add_argument('--trojan_weight_decay', type=float, default=0.0,
                    help='weight decay of trojan network')
parser.add_argument('--discrete_thrd', type=float, default=0.5,)
parser.add_argument('--if_load_watermark', action='store_true', default=False,
                    help='Load watermarked encoder pretrained in ZINC dataset.')
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
    pretrain_dataset = MoleculeDataset(dataset_root + dataset_name, dataset=dataset_name)
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
    dataset = MoleculeDataset(dataset_root + dataset_name, dataset=dataset_name)
    '''sample only 200k graphs from ZINC'''
    sampled_idx = utils.obtain_idx_random(args.seed, np.array(list(range(len(dataset)))), size = int(len(dataset) * 0.1))
    dataset = dataset[sampled_idx]
elif(args.dataset == 'MNIST' or args.dataset == 'CIFAR10'):
    dataset = GNNBenchmarkDataset(root='./data/', name=args.dataset)
dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True)

#%%

from models.IPGCL import Encoder
from models.Backbones import GIN
import GCL.augmentors as A
import GCL.losses as L
import torch.nn as nn
from GCL.models import DualBranchContrast

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

contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
gcl = Encoder(encoder=gnn,augmentor=(aug1,aug2),project=project, contrast_model=contrast_model,device=device).to(device)
gcl_init = copy.deepcopy(gcl)
gcl_shadow = copy.deepcopy(gcl)
backdoor_gcl = copy.deepcopy(gcl)

# select backdoored graphs
from models.GCBA_poi import GCBA

pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=args.batch_size)
# print("Pretrain the encoder without watermarking")
# gcl.pretrain(dataloader, args)
gcl.encoder.load_state_dict(torch.load("./checkpoint/{}_GCL/gcl_{}.pt".format('ZINC200k', 10)))
gcl_shadow = copy.deepcopy(gcl)
print("Learn to create poison data")
trigger_model = GCBA(args, device)
poison_pretrain_dataset, _ = trigger_model.fit(pretrain_dataset, gcl_shadow)
print("Pretrain Poisoned Encoder")
poison_pretrain_dataloader = DataLoader(poison_pretrain_dataset, batch_size=args.batch_size)

if(args.if_load_watermark):
    backdoor_gcl.encoder.load_state_dict(torch.load("./checkpoint/{}_GCL/bkd_gcl_{}.pt".format('ZINC200k', 10)))
else:
    backdoor_gcl.pretrain(poison_pretrain_dataloader, args)
    torch.save(backdoor_gcl.encoder.state_dict(), "./checkpoint/{}_GCL/bkd_gcl_{}.pt".format('ZINC200k', 10))


#%%
# select normal graphs
clean_acc_plains, asr_plains = [], []
clean_acc_bkds, asr_bkds = [], []
asr_false_plains = []
asr_false_bkds = []
asr_diffs = []
for seed in range(10,10+args.num_repeat):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    split_train = utils.get_split_self(num_samples=len(dataset), train_ratio=0.5, test_ratio=0.4,seed=seed,device=device)
    idx_train = split_train['train']    
    idx_test = split_train['test']
    idx_val = split_train['valid']
    idx_clean_test = split_train['clean_test']
    idx_atk = split_train['clean_test']

    # idx_bkd_candidate = []
    # for i in range(len(dataset)):
    #     if dataset[i].y.item() != args.target_class:
    #         idx_bkd_candidate.append(i)
    # idx_bkd_candidate = np.array(idx_bkd_candidate)

    # rs = np.random.RandomState(42)
    # rs.shuffle(idx_bkd_candidate)

    # num_clean_test = int(len(dataset) * 0.2)
    # num_atk = int(len(dataset) * 0.2)
    # idx_clean_test = idx_bkd_candidate[:num_clean_test]
    # idx_atk = idx_bkd_candidate[num_clean_test:num_clean_test+num_atk]

    from torch.utils.data import Subset
    trainloader = DataLoader(Subset(dataset,idx_train), batch_size=args.batch_size)
    testloader = DataLoader(Subset(dataset,idx_test), batch_size=args.batch_size)
    valloader = DataLoader(Subset(dataset,idx_val),batch_size=args.batch_size)

    '''after pretraining, finetune encoder in the downstream dataset'''
    ft_gcl = copy.deepcopy(gcl)
    ft_gcl.transfer_finetune(trainloader, testloader, valloader, args, num_tasks, verbose=args.debug)

    ft_bkd_gcl = copy.deepcopy(backdoor_gcl)
    ft_bkd_gcl.transfer_finetune(trainloader, testloader, valloader, args, num_tasks, verbose=args.debug)


    print("Before Backdooring")
    clean_acc_plain, asr_plain, asr_false_plain = trigger_model.test(ft_gcl, dataset, idx_train, idx_test, idx_atk)
    clean_acc_bkd, asr_bkd, asr_false_bkd = trigger_model.test(ft_bkd_gcl, dataset, idx_train, idx_test, idx_atk)
    clean_acc_plains.append(clean_acc_plain)
    asr_plains.append(asr_plain)
    clean_acc_bkds.append(clean_acc_bkd)
    asr_bkds.append(asr_bkd)
    asr_false_plains.append(asr_false_plain)
    asr_false_bkds.append(asr_false_bkd)

    asr_diff = asr_bkd - asr_plain
    asr_diffs.append(asr_diff)


print("==============Plain=================")
print("Clean acc: {:.4f}+-{:.4f}, ASR: {:.4f}+-{:.4f}".format(np.mean(clean_acc_plains), np.std(clean_acc_plains), np.mean(asr_plains), np.std(asr_plains)))
print("ASR false: {:.4f}+-{:.4f}".format(np.mean(asr_false_plains), np.std(asr_false_plains)))
print("==============Backdoor=================")
print("Clean acc: {:.4f}+-{:.4f}, ASR: {:.4f}+-{:.4f}".format(np.mean(clean_acc_bkds), np.std(clean_acc_bkds), np.mean(asr_bkds), np.std(asr_bkds)))
print("ASR false: {:.4f}+-{:.4f}".format(np.mean(asr_false_bkds), np.std(asr_false_bkds)))
print("=============IP Evaluation=============")
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score


y = np.concatenate([np.zeros(args.num_repeat), np.ones(args.num_repeat)]) 
score_fix = roc_auc_score(y,  np.concatenate([asr_plains, asr_bkds]))
diff_fix = np.asarray(asr_bkds) - np.asarray(asr_plains)
print("ROC score (Fixed): {:.4f}".format(score_fix))
print("p-value: {}, Difference: {:.4f}, {:.4f}"\
      .format(score_fix,ttest_rel(asr_plains, asr_bkds).pvalue, np.mean(diff_fix), np.std(diff_fix)))

# gcl.test(dataloader,idx_train,idx_test,key_normal, key_watermark,args)
# # args.num_epochs = 200
# print("Watermarking")
# gcl = copy.deepcopy(gcl_init)
# gcl.watermarking(key_normal, key_watermark, dataloader, args, verbose=args.debug)
# gcl.test(dataloader,idx_train,idx_test,key_normal, key_watermark,args)

# acc_list = []
# IP_list = []
# for seed in range(10,20):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     # print("\n==============New test=================")
#     acc, IP = gcl.test_finetune(trainloader,testloader,key_normal, key_watermark,args)
#     acc_list.append(acc)
#     IP_list.append(IP)
 

# print("Finetune acc: {:.4f}, {:.4f}, IP_acc: {:.4f}, {:.4f}"\
#       .format(np.mean(acc_list),np.std(acc_list), np.mean(IP_list), np.std(IP_list)))


# %%
