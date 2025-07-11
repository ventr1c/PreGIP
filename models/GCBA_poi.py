import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import copy
from torch_geometric.data import Data, Batch, DataLoader
import torch.optim as optim
from utils import accuracy

def get_ref_idx(dataset, select_number, target_class):
    bkd_idx = []
    for i in range(len(dataset)):
        if dataset[i].y == target_class:
            bkd_idx.append(i)
    rs = np.random.RandomState(10)
    rs.shuffle(bkd_idx)
    bkd_idx = bkd_idx[:int(select_number)]
    return bkd_idx

def random_get_idx(dataset, select_number, idx_exists = None):
    if(idx_exists is not None):
        bkd_idx = list(range(len(dataset)))
        bkd_idx = list(set(bkd_idx) - set(idx_exists))
        rs = np.random.RandomState(10)
        rs.shuffle(bkd_idx)
        bkd_idx = bkd_idx[:int(select_number)]
        return bkd_idx
    else:
        bkd_idx = []
        bkd_idx = list(range(len(dataset)))
        rs = np.random.RandomState(10)
        rs.shuffle(bkd_idx)
        bkd_idx = bkd_idx[:int(select_number)]
        return bkd_idx
class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat,nout*nfeat)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight

class GCBA(nn.Module):
    def __init__(self, args, device):
        super(GCBA, self).__init__()
        self.args = args
        self.device = device
        self.weights = None
        self.init_v = 0
        self.trigger_index = self.get_trigger_index(self.args.trigger_size)
        
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    # def set_temp(self, temp):
    #     self.temp = temp

    def get_trigger_index(self,trigger_size):
        edge_list = []
        edge_list.append([0,0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T
        return edge_index
    
    def get_trojan_edge(self,start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index 
    
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        mean_x = torch.unsqueeze(torch.mean(features,dim=0),dim=0)
        trojan_feat, trojan_weights = self.trojan(mean_x,self.args.discrete_thrd) # may revise the process of generate
        
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights
    
    def test(self, model, dataset, idx_train, idx_clean_test, idx_atk):
        model.eval()
        self.trojan.eval()

        # clean_dataset = copy.deepcopy(dataset)
        # h_poi_clr = self.h_poisons[self.idx_clr]
        
        '''Clean Accuracy'''
        x_clean = []
        y_clean = []
        
        clean_dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        clean_train_dataloader = DataLoader(dataset[idx_train], batch_size=self.args.batch_size, shuffle=False)
        clean_test_dataloader = DataLoader(dataset[idx_clean_test], batch_size=self.args.batch_size, shuffle=False)
        with torch.no_grad():
            for data in clean_dataloader:
                data = data.to(self.device)
                _, g = model.encoder(data.x, data.edge_index, data.edge_attr, data.batch, edge_weight=data.edge_weight)
                x_clean.append(g.detach())
                y_clean.append(data.y)
            x_clean = torch.cat(x_clean, dim=0)
            y_clean = torch.cat(y_clean, dim=0)


        # linear evaluation
        num_classes = dataset.num_classes
        fc = nn.Linear(x_clean.shape[1], num_classes).to(self.device)
        optimizer = torch.optim.Adam(fc.parameters(), lr=0.001, weight_decay=self.args.weight_decay)

        for _ in range(1000):
            optimizer.zero_grad()
            pred = fc(x_clean[idx_train])
            loss = torch.nn.functional.cross_entropy(pred,y_clean[idx_train])
            loss.backward()
            optimizer.step()

        pred_test = torch.softmax(fc(x_clean[idx_clean_test]),dim=1)
        clean_acc = accuracy(pred_test,y_clean[idx_clean_test])
        print("CLean Accuracy: {:.4f}".format(clean_acc))


        '''Backdoor ASR'''
        x_atk = []
        y_atk = []
         
        atk_dataset = []
        for idx in idx_atk:
            data = dataset[idx].to(self.device)
            if(data.edge_weight is None):
                data.edge_weight = torch.ones([data.edge_index.shape[1]],device=self.device)
            mean_x = torch.unsqueeze(torch.mean(data.x,dim=0),dim=0)
            '''update poison_edge_index'''
            idx_attach = torch.randint(0, len(data.x), (1,)).to(self.device)
            bkd_x, bkd_edge_index, bkd_edge_weight = self.inject_trigger(idx_attach, data.x, data.edge_index, data.edge_attr, data.edge_weight, self.device)
            bkd_x, bkd_edge_index, bkd_edge_weight = bkd_x.clone().detach(), bkd_edge_index.clone().detach(), bkd_edge_weight.clone().detach()
            
            bkd_data = Data(x=bkd_x, edge_index=bkd_edge_index, y=data.y, edge_weight=bkd_edge_weight)
            atk_dataset.append(bkd_data)
        atk_dataloader = Batch.from_data_list(atk_dataset)

        bkd_dataloader = DataLoader(atk_dataset, batch_size=self.args.batch_size, shuffle=False)
        with torch.no_grad():
            for data in bkd_dataloader:
                data = data.to(self.device)
                _, g = model.encoder(data.x, data.edge_index, data.edge_attr, data.batch, edge_weight=data.edge_weight)
                x_atk.append(g.detach())
                y_atk.append(data.y)
            x_atk = torch.cat(x_atk, dim=0)
            y_atk = torch.cat(y_atk, dim=0)

        # linear evaluation
        num_classes = dataset.num_classes
        fc = nn.Linear(x_atk.shape[1], num_classes).to(self.device)
        optimizer = torch.optim.Adam(fc.parameters(), lr=0.001, weight_decay=self.args.weight_decay)

        for _ in range(1000):
            optimizer.zero_grad()
            pred = fc(x_atk)
            loss = torch.nn.functional.cross_entropy(pred,y_atk)
            loss.backward()
            optimizer.step()

        pred_atk = torch.softmax(fc(x_atk),dim=1)
        atk_asr = (pred_atk.argmax(dim=1)==self.args.target_class).float().mean()
        atk_asr_no_true = (pred_atk.argmax(dim=1)!=y_atk).float().mean()
        print("Attack Success Rate: {:.4f}".format(atk_asr))
        print("Attack Success Rate without True Class: {:.4f}".format(atk_asr_no_true))
        return clean_acc.item(), atk_asr.item(), atk_asr_no_true.item()

    def fit(self, dataset, model):
        self.model_only_epoch = 30
        bkd_dataset = copy.deepcopy(dataset) 
        target_class = self.args.target_class
        ori_dataset = copy.deepcopy(dataset)
        feature_dim = dataset[0].x.shape[1]
        self.idx_ref = get_ref_idx(dataset, self.args.ref_number, target_class)
        self.idx_bkd = random_get_idx(dataset, self.args.bkd_number, idx_exists = self.idx_ref)
        self.idx_clr = random_get_idx(dataset, self.args.clr_number, idx_exists = self.idx_bkd)
        # self.theta = nn.Parameter(torch.ones_like(dataset[0].x[0])*self.init_v, requires_grad=True)

        # initialize a shadow GCL model
        self.target_model = copy.deepcopy(model) 
        self.target_model.to(self.device)
        self.target_model.train()
        # initialize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, feature_dim, self.args.trigger_size, layernum=2).to(self.device)
        self.trojan.train()
        optimizer_target = optim.Adam(self.target_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.trojan_lr, weight_decay=self.args.trojan_weight_decay)

        if(self.args.feat_contineous):
            # attach trigger into the graphs
            bkd_dataset = copy.deepcopy(dataset)
            bkd_dataset = []
            ori_edge_weights = []
            
            # for data in dataset:
            #     data = data.to(self.device)
            #     edge_weight = torch.ones([data.edge_index.shape[1]],device=self.device)
            #     ori_edge_weights.append(edge_weight)
            #     '''randomly attach the trigger into current graph'''
            #     idx_attach = torch.randint(0, len(data.x), (1,)).to(self.device)
            #     trojan_edge = self.get_trojan_edge(len(data.x),idx_attach,self.args.trigger_size).to(self.device)
            #     poison_edge_index = torch.cat([data.edge_index,trojan_edge],dim=1)
            #     # trojan_feat = torch.zeros([self.args.trigger_size, data.x.shape[1]], device=self.device)
            #     # poison_x = torch.cat([data.x,trojan_feat],dim=0)
            #     poison_x = data.x.clone()
            #     bkd_data = Data(x=poison_x, edge_index=poison_edge_index, y=data.y)
            #     bkd_dataset.append(bkd_data)

            for epoch in range(self.args.trojan_epochs):
                self.trojan.train()

                for j in range(1):
                    # optimizer_trigger.zero_grad()
                    # if(self.args.feat_contineous):
                    #     self.trojan.set_temp(2)
                    h_oris = torch.FloatTensor([]).to(self.device)
                    h_poisons = torch.FloatTensor([]).to(self.device)
                    poison_dataset = []
                    l3 = 0
                    loss_inner = 0
                    for i in range(len(dataset)):
                        # bkd_data = bkd_dataset[i].to(self.device)
                        ori_data = ori_dataset[i].to(self.device)
                        ori_edge_weight = torch.ones([ori_data.edge_index.shape[1]],device=self.device)
                        # ori_edge_weights.append(ori_edge_weight)

                        _, h_ori = self.target_model.encoder(ori_data.x, ori_data.edge_index, ori_data.edge_attr, batch = None, edge_weight = None)
                        h_oris = torch.cat([h_oris,h_ori],dim=0)
                        mean_x = torch.unsqueeze(torch.mean(ori_data.x,dim=0),dim=0)

                        '''update poison_edge_index'''
                        idx_attach = torch.randint(0, len(ori_data.x), (1,)).to(self.device)
                        trojan_edge = self.get_trojan_edge(len(ori_data.x),idx_attach,self.args.trigger_size).to(self.device)
                        poison_edge_index = torch.cat([ori_data.edge_index,trojan_edge],dim=1)

                        '''update poison_edge_weight and poison_x'''
                        trojan_feat, trojan_weights = self.trojan(mean_x,self.args.discrete_thrd)
                        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                        trojan_weights = trojan_weights.flatten()
                        trojan_feat = trojan_feat.view([-1,ori_data.x.shape[1]])

                        poison_edge_weights = torch.cat([ori_edge_weight,trojan_weights,trojan_weights])
                        poison_x = torch.cat([ori_data.x,trojan_feat]).detach()

                        # l3 = torch.sum(poison_x[-1]*(1-poison_x[-1]))

                        '''get poisoned embedding'''
                        poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=ori_data.y, edge_weight=poison_edge_weights)
                        # cont_loss = self.target_model.contrastive_loss(poison_data)
                        # loss_inner += cont_loss
                        poison_dataset.append(poison_data)
                        poison_data = Batch.from_data_list([poison_data])
                        _, h_poison = self.target_model.encoder(poison_data.x, poison_data.edge_index, poison_data.edge_attr, batch = None, edge_weight = poison_data.edge_weight)
                        h_poisons = torch.cat([h_poisons,h_poison],dim=0)
                    h_mean_poi_ref = torch.unsqueeze(torch.mean(h_poisons,dim=0),dim=0)
                    h_mean_poi_ref = h_mean_poi_ref.repeat(len(self.idx_ref),1)
                    h_poi_bkd = h_poisons[self.idx_bkd]
                    
                    # poison_dataloader = DataLoader(poison_dataset, batch_size = self.args.batch_size)
                    # for data in poison_dataloader:
                    #     data = data.to(self.device)
                    #     cont_loss = self.target_model.contrastive_loss(data)
                    #     loss_inner = cont_loss
                    #     optimizer_target.zero_grad()
                    #     loss_inner.backward()
                    #     optimizer_target.step()
                # h_mean_poi_ref = torch.unsqueeze(torch.mean(h_poisons,dim=0),dim=0)
                # h_mean_poi_ref = h_mean_poi_ref.repeat(len(self.idx_ref),1)
                # h_poi_bkd = h_poisons[self.idx_bkd]

                l0 = -torch.mean(self.cos_sim(h_poi_bkd, h_mean_poi_ref))
                l1 = -torch.mean(self.cos_sim(h_poisons[self.idx_ref], h_oris[self.idx_ref]))
                l2 = -torch.mean(self.cos_sim(h_poisons[self.idx_clr], h_oris[self.idx_clr]))
                lambda3 = 0
                lambda1 = 1
                lambda2 = 1
                loss_outter =  l0 + lambda1*l1 + lambda2*l2
                # self.trojan.eval()
                optimizer_target.zero_grad()
                if(epoch < self.model_only_epoch):
                    optimizer_trigger.zero_grad()

                loss_outter.backward()
                optimizer_target.step()
                if(epoch < self.model_only_epoch):
                    optimizer_trigger.step()
                if((epoch+1) % 5 == 0):
                    print("Epoch: {}, {:.4f} {:.4f} {:.4f}".format(epoch, l0.detach().cpu().numpy().item(),l1.detach().cpu().numpy().item(),l2.detach().cpu().numpy().item()))
            #     h_oris = torch.FloatTensor([]).to(self.device)
            #     h_poisons = torch.FloatTensor([]).to(self.device)
            #     poison_dataset = []
            #     for i in range(len(dataset)):
            #         # bkd_data = bkd_dataset[i].to(self.device)
            #         ori_data = ori_dataset[i].to(self.device)
            #         ori_edge_weight = torch.ones([ori_data.edge_index.shape[1]],device=self.device)
            #         # ori_edge_weights.append(ori_edge_weight)

            #         _, h_ori = self.target_model.encoder(ori_data.x, ori_data.edge_index, batch = None, edge_weight = None)
            #         h_oris = torch.cat([h_oris,h_ori],dim=0)
            #         mean_x = torch.unsqueeze(torch.mean(ori_data.x,dim=0),dim=0)

            #         '''update poison_edge_index'''
            #         idx_attach = torch.randint(0, len(ori_data.x), (1,)).to(self.device)
            #         trojan_edge = self.get_trojan_edge(len(ori_data.x),idx_attach,self.args.trigger_size).to(self.device)
            #         poison_edge_index = torch.cat([ori_data.edge_index,trojan_edge],dim=1)

            #         '''update poison_edge_weight and poison_x'''
            #         trojan_feat, trojan_weights = self.trojan(mean_x,self.args.discrete_thrd)
            #         trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
            #         trojan_weights = trojan_weights.flatten()
            #         trojan_feat = trojan_feat.view([-1,ori_data.x.shape[1]])

            #         poison_edge_weights = torch.cat([ori_edge_weight,trojan_weights,trojan_weights])
            #         poison_x = torch.cat([ori_data.x,trojan_feat])

            #         l3 = torch.sum(poison_x[-1]*(1-poison_x[-1]))

            #         '''get poisoned embedding'''
            #         poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=ori_data.y, edge_weight=poison_edge_weights)
            #         poison_dataset.append(poison_data)
            #         poison_data = Batch.from_data_list([poison_data])
            #         _, h_poison = self.target_model.encoder(poison_data.x, poison_data.edge_index, batch = None, edge_weight = poison_data.edge_weight)
            #         h_poisons = torch.cat([h_poisons,h_poison],dim=0)
            #     h_mean_poi_ref = torch.unsqueeze(torch.mean(h_poisons,dim=0),dim=0)
            #     h_mean_poi_ref = h_mean_poi_ref.repeat(len(self.idx_ref),1)
            #     h_poi_bkd = h_poisons[self.idx_bkd]

            #     l0 = -torch.mean(self.cos_sim(h_poi_bkd, h_mean_poi_ref))
            #     l1 = -torch.mean(self.cos_sim(h_poisons[self.idx_ref], h_oris[self.idx_ref]))
            #     l2 = -torch.mean(self.cos_sim(h_poisons[self.idx_clr], h_oris[self.idx_clr]))
            #     lambda3 = 0
            #     lambda1 = 1
            #     lambda2 = 1
            #     loss_outter =  l0 + lambda1*l1 + lambda2*l2
            #     # self.trojan.eval()
            #     optimizer_target.zero_grad()
            #     loss_outter.backward()
            #     optimizer_target.step()
            #     # self.trojan.train()
            #     # for j in range(self.args.inner):
            #     #     pass
            #     #     optimizer_target.zero_grad()
            #     if(epoch % 5 == 0):
            #         print("Epoch: {}, {:.4f} {:.4f} {:.4f} {:.4f}".format(epoch, loss_inner.detach().cpu().numpy().item(), l0.detach().cpu().numpy().item(),l1.detach().cpu().numpy().item(),l2.detach().cpu().numpy().item()))
            #     # print("Epoch: {}, {:.4f} {:.4f} {:.4f}".format(epoch, l0.detach().cpu().numpy().item(),l1.detach().cpu().numpy().item(),l2.detach().cpu().numpy().item()))
            # # self.h_poisons = h_poisons

            poison_dataset = []
            for i in range(len(dataset)):
                # bkd_data = bkd_dataset[i].to(self.device)
                ori_data = ori_dataset[i].to(self.device)
                ori_edge_weight = torch.ones([ori_data.edge_index.shape[1]],device=self.device)
                # ori_edge_weights.append(ori_edge_weight)
                mean_x = torch.unsqueeze(torch.mean(ori_data.x,dim=0),dim=0)

                '''update poison_edge_index'''
                idx_attach = torch.randint(0, len(ori_data.x), (1,)).to(self.device)
                trojan_edge = self.get_trojan_edge(len(ori_data.x),idx_attach,self.args.trigger_size).to(self.device)
                poison_edge_index = torch.cat([ori_data.edge_index,trojan_edge],dim=1)

                '''update poison_edge_weight and poison_x'''
                trojan_feat, trojan_weights = self.trojan(mean_x,self.args.discrete_thrd)
                trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1,ori_data.x.shape[1]])

                poison_edge_weights = torch.cat([ori_edge_weight,trojan_weights,trojan_weights])
                poison_x = torch.cat([ori_data.x,trojan_feat])


                '''get poisoned embedding'''
                poison_data = Data(x=poison_x.detach(), edge_index=poison_edge_index, y=ori_data.y, edge_weight=poison_edge_weights)
                # cont_loss = self.target_model.contrastive_loss(poison_data)
                poison_dataset.append(poison_data)
            return poison_dataset, self.target_model
        else:
            # self.theta = nn.Parameter(torch.ones_like(dataset[0].x)*self.init_v, requires_grad=True)
            raise NotImplementedError
