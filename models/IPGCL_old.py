import torch
import torch.nn as nn
import torch.nn.functional as F
from GCL.eval import get_split, LREvaluator, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.data import DataLoader
from utils import accuracy
import copy
from utils import parameter_diff, clone_module, update_module
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, project, contrast_model, device):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.project = project
        self.contrast_model = contrast_model
        self.device = device

    def forward(self, x, edge_index, batch, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)

        z1, g1 = self.encoder(x1, edge_index1, batch, edge_weight)
        z2, g2 = self.encoder(x2, edge_index2, batch, edge_weight)
        return z1, z2, g1, g2

    def reset_model_parameter(self):
        self.encoder.reset_parameters()
        self.project.reset_parameters()
        self.contrast_model.reset_parameters()

    def contrastive_loss(self,data):

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, g1, g2 = self.forward(data.x, data.edge_index, data.batch)
        g1, g2 = [self.project(g) for g in [g1, g2]]
        loss = self.contrast_model(g1=g1, g2=g2, batch=data.batch)

        return loss

    def pretrain(self, dataloader, args,verbose=True):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.train()
        for epoch in range(args.num_epochs):

            epoch_loss = 0

            for data in dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()

                loss = self.contrastive_loss(data)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            epoch_loss = epoch_loss/len(dataloader)
            if verbose and epoch % 5 == 0:
                print('Epoch {}, training loss: {}'.format(epoch,epoch_loss))

    def transfer_eval(self, encoder, fc, test_dataloader):
        encoder.eval()
        y_true = []
        y_scores = []
        for data in test_dataloader:
            pass
            data = data.to(self.device)
            with torch.no_grad():
                _, g = encoder(data.x, data.edge_index, data.batch)
                pred = fc(g)
            y_true.append(data.y.view(pred.shape))
            y_scores.append(pred)
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

        roc_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

        return sum(roc_list)/len(roc_list) #y_true.shape[1]

    def transfer_finetune(self, train_dataloader, test_dataloader, val_dataloader, args, num_tasks, verbose=True):
        # optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss(reduction = "none")
        model = copy.deepcopy(self.encoder)
        '''define prediction linear layer'''
        fc = nn.Linear(args.num_hidden, num_tasks).to(self.device)
        optimizer_fc = torch.optim.Adam(fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_enc = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


        for epoch in range(args.num_epochs):
            model.train()
            for data in train_dataloader:

                optimizer_fc.zero_grad()
                optimizer_enc.zero_grad()
                data = data.to(self.device)
                _, g = model(data.x, data.edge_index, data.batch)
                pred = fc(g)
                y = data.y.view(pred.shape).to(torch.float64)
                #Whether y is non-null or not.
                is_valid = y**2 > 0
                loss_mat = criterion(pred.double(), (y+1)/2)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat)/torch.sum(is_valid)

                # loss = torch.nn.functional.cross_entropy(pred,data.y)
                loss.backward()
                optimizer_fc.step()
                optimizer_enc.step()

            if verbose and (epoch+1) % 5 == 0:
                print('Epoch {}, training loss: {}'.format(epoch,loss))
                
                if(args.eval_train):
                    train_acc = self.transfer_eval(model, fc, train_dataloader)
                else:
                    print("omit the training accuracy computation")
                    train_acc = 0
                val_acc = self.transfer_eval(model, fc, val_dataloader)
                test_acc = self.transfer_eval(model, fc, test_dataloader)
                print('Epoch {}, train acc: {}, val acc: {}, test acc: {}'.format(epoch, train_acc, val_acc, test_acc))

        return model, fc

    def watermark_loss(self, model, key_normal, key_watermark, data):

        _,normal_h = model(key_normal.x,key_normal.edge_index,key_normal.batch)
        _,watermark_h = model(key_watermark.x, key_watermark.edge_index, key_watermark.batch)
        _, other_h = model(data.x, data.edge_index, data.batch)
        score = nn.functional.pairwise_distance(normal_h,watermark_h).mean()
        watermark_loss = score + torch.clamp_min(1.0 + score.detach() - torch.cdist(other_h,watermark_h).mean(),0.0)

        return watermark_loss
    


        

    def meta_IP(self, key_normal, key_watermark, data, eps, step, random):
        """
        eps: the ball of finding the worst updates of model parameters
        step: how many steps we apply here
        second_order: whether compute the second order graident
        """

        clone = clone_module(self.encoder)
        
        meta_loss = 0.0
        for _ in range(step):
            wm_loss = self.watermark_loss(clone, key_normal,key_watermark,data)

            if random:
                grads = [2*torch.rand_like(g)-1 for g in clone.parameters()]
                
            else:
                grads = torch.autograd.grad(wm_loss, clone.parameters(), retain_graph=True)

            total_norm = torch.norm(torch.stack([torch.norm(g, p=2) for g in grads]), p=2)

            if total_norm < 1.0:
                total_norm = 1.0
            lr = eps/(float(step)*total_norm)

            updates = [lr * g for g in grads] # here apply gradient ascent
            clone = update_module(clone, updates)

            meta_loss = meta_loss + wm_loss

        meta_loss = (meta_loss+self.watermark_loss(clone, key_normal,key_watermark,data))/(step+1)
        return meta_loss




    def watermarking(self, key_normal, key_watermark, dataloader, args, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.train()
        for epoch in range(args.num_epochs):

            epoch_loss = 0

            for data in dataloader:
                data = data.to(self.device)
                key_normal = key_normal.to(self.device)
                key_watermark = key_watermark.to(self.device)

                optimizer.zero_grad()

                cont_loss = self.contrastive_loss(data)
                wm_loss = self.watermark_loss(self.encoder, key_normal, key_watermark, data)
                meta_loss = self.meta_IP(key_normal, key_watermark, data, eps=args.eps, step=args.step, random=args.random)
                loss = cont_loss + args.alpha *meta_loss

                loss.backward()
                optimizer.step()

                epoch_loss += cont_loss.item()
            epoch_loss = epoch_loss/len(dataloader)

            if verbose and (epoch+1) % 5 == 0:
                print('Epoch {}, contrastive loss: {:.6f}, Watermark loss: {:.6f}, Meta IP loss: {:.6f}'\
                      .format(epoch+1,epoch_loss, wm_loss.item(), meta_loss.item()))
                
        print('Epoch {}, contrastive loss: {:.6f}, Watermark loss: {:.6f}, Meta IP loss: {:.6f}'\
                .format(epoch+1,epoch_loss, wm_loss.item(), meta_loss.item()))
        

    def test(self, train_loader,test_loader, key_normal, key_watermark, args):
        self.eval()
        x_train = []
        y_train = []

        with torch.no_grad():
            for data in train_loader:
                data = data.to(self.device)
                print(data.y)
                _, g = self.encoder(data.x, data.edge_index, data.batch)
                x_train.append(g.detach())
                y_train.append(data.y)
            x_train = torch.cat(x_train, dim=0)
            y_train = torch.cat(y_train, dim=0)

        num_classes = y_train.max().item() + 1
        fc = nn.Linear(x_train.shape[1], num_classes).to(self.device)
        optimizer = torch.optim.Adam(fc.parameters(), lr=0.001, weight_decay=args.weight_decay)

        for _ in range(1000):
            optimizer.zero_grad()
            pred = fc(x_train)
            loss = torch.nn.functional.cross_entropy(pred,y_train)
            loss.backward()
            optimizer.step()


        x_test = []
        y_test = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                _, g = self.encoder(data.x, data.edge_index, data.batch)
                x_test.append(g.detach())
                y_test.append(data.y)
            x_test = torch.cat(x_test, dim=0)
            y_test = torch.cat(y_test, dim=0)

        pred_test = torch.softmax(fc(x_test),dim=1)
       
        acc = accuracy(pred_test,y_test)
        
        if(key_normal == None or key_watermark == None):
            if args.debug:
                print("Accuracy: {:.4f}".format(acc))
            return float(acc)
        else:
            key_normal = key_normal.to(self.device)
            key_watermark = key_watermark.to(self.device)
            _,normal_h = self.encoder(key_normal.x,key_normal.edge_index,key_normal.batch)
            _,watermark_h = self.encoder(key_watermark.x, key_watermark.edge_index, key_watermark.batch)

            score_normal, pred_normal = torch.softmax(fc(normal_h.detach()),dim=1).max(1)
            # print(pred_normal)
            score_watermark, pred_watermark = torch.softmax(fc(watermark_h.detach()),dim=1).max(1)
            # print(pred_watermark)
            result = torch.concat([pred_normal,pred_watermark])
            sign_score = torch.eye(num_classes,device=self.device)[result].float().mean(dim=0).max()
            # print("sing score is {:.4f}".format(sign_score))

            if args.debug:
                print("normal confidence score: {:.4f}, {:.4f}, watermark confidence score: {:.4f}, {:.4f}"\
                    .format(score_normal.mean().item(), score_normal.std().item(), score_watermark.mean().item(), score_watermark.std().item()))
            correct = pred_normal.eq(pred_watermark).double()
            IP_acc = correct.sum()/len(pred_watermark)

            if args.debug:
                print("Accuracy: {:.4f}, IP_ACC: {:.4f}".format(acc, IP_acc))

            return float(acc), float(IP_acc)
    

    def test_finetune(self, train_loader,test_loader, key_normal, key_watermark, args):
        
        refer = self.encoder
        model = copy.deepcopy(self.encoder)
        num_classes = train_loader.dataset.dataset.data.y.max().item() + 1
        fc = nn.Linear(args.num_hidden, num_classes).to(self.device)
        optimizer_fc = torch.optim.Adam(fc.parameters(), lr=0.0003, weight_decay=args.weight_decay)
        optimizer_enc = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=args.weight_decay)


        for epoch in range(args.test_iteration):
            model.train()
            for data in train_loader:

                optimizer_fc.zero_grad()
                optimizer_enc.zero_grad()
                data = data.to(self.device)
                _, g = model(data.x, data.edge_index, data.batch)
                pred = fc(g)

                loss = torch.nn.functional.cross_entropy(pred,data.y)
                loss.backward()
                optimizer_fc.step()
                optimizer_enc.step()

            w_diff = float(parameter_diff(model.parameters(), refer.parameters()).detach())

            key_normal = key_normal.to(self.device)
            key_watermark = key_watermark.to(self.device)
            _,normal_h = model(key_normal.x,key_normal.edge_index,key_normal.batch)
            _,watermark_h = model(key_watermark.x, key_watermark.edge_index, key_watermark.batch)


            score_normal, pred_normal = torch.softmax(fc(normal_h.detach()),dim=1).max(1)
            score_watermark, pred_watermark = torch.softmax(fc(watermark_h.detach()),dim=1).max(1)


            correct = pred_normal.eq(pred_watermark).double()
            IP_acc = correct.sum()/len(pred_watermark)


            model.eval()
            pred = []
            y = []
            with torch.no_grad():
                for data in test_loader:

                    data = data.to(self.device)
                    _, g = model(data.x, data.edge_index, data.batch)

                    pred.append(fc(g))
                    y.append(data.y)

            pred = torch.cat(pred, dim=0)
            y = torch.cat(y, dim=0)
            acc = accuracy(pred,y)

            if args.debug:
                print("Epoch: {}, Fine-tune Acc: {:.4f}, IP_ACC: {:.4f}, w_diff: {}"\
                    .format(epoch, acc, IP_acc, w_diff))
                
        result = torch.concat([pred_normal,pred_watermark])
        sign_score = torch.eye(num_classes,device=self.device)[result].float().mean(dim=0).max()
        # print("sing score is {:.4f}".format(sign_score))
        if args.debug:
            print("normal confidence score: {:.4f}, {:.4f}, watermark confidence score: {:.4f}, {:.4f}"\
                .format(score_normal.mean().item(), score_normal.std().item(), score_watermark.mean().item(), score_watermark.std().item()))

        return float(acc), float(IP_acc)

