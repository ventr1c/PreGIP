import torch
import torch.nn as nn
from utils import accuracy
import copy
from utils import parameter_diff, clone_module, update_module
import numpy as np
from sklearn.cluster import KMeans
<<<<<<< HEAD
from sklearn.metrics import roc_auc_score
from copy import deepcopy
=======
>>>>>>> b99538ad237ca2f0286619ea31de9e82c9204f15

class Encoder(torch.nn.Module):
    def __init__(self, args, encoder, project, device):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = project
        self.device = device
        self.args = args
        self.linear_pred_atoms = torch.nn.Linear(self.args.num_hidden, 120).to(device)
        self.linear_pred_bonds = torch.nn.Linear(self.args.num_hidden, 6).to(device)

    def pretrain(self, dataloader, args,verbose=True):
        criterion = nn.CrossEntropyLoss()
        #set up optimizers
        # model_list = [self.encoder, self.linear_pred_atoms, self.linear_pred_bonds]

        optimizer_model = torch.optim.Adam(self.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_linear_pred_atoms = torch.optim.Adam(self.linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_linear_pred_bonds = torch.optim.Adam(self.linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]
        self.encoder.train()
        self.linear_pred_atoms.train()
        self.linear_pred_bonds.train()

        for epoch in range(args.num_epochs):
            loss_accum = 0
            acc_node_accum = 0
            acc_edge_accum = 0
            for data in dataloader:
                data = data.to(self.device)
                optimizer_model.zero_grad()
                optimizer_linear_pred_atoms.zero_grad()
                optimizer_linear_pred_bonds.zero_grad()

                loss, acc_node, acc_edge = self.pred_loss(data)
                acc_node_accum += acc_node
                acc_edge_accum += acc_edge

                loss.backward()

                optimizer_model.step()
                optimizer_linear_pred_atoms.step()
                optimizer_linear_pred_bonds.step()

                loss_accum += float(loss.cpu().item())
            epoch_loss = loss_accum/len(dataloader)
            epoch_acc_node = acc_node_accum/len(dataloader)
            epoch_acc_edge = acc_edge_accum/len(dataloader)
            if verbose and epoch % 5 == 0:
                print('Epoch {}, training loss: {}, node acc: {}, edge acc: {}'.format(epoch,epoch_loss,epoch_acc_node,epoch_acc_edge))

    def pred_loss(self,data):
        criterion = nn.CrossEntropyLoss()
        node_rep, g = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        ## loss for nodes
        # print(data.masked_atom_indices)
        # print(node_rep.shape)
        pred_node = self.linear_pred_atoms(node_rep[data.masked_atom_indices])
        # print(pred_node.shape)
        # print(data.mask_node_label[:,0].unique())
        pred_loss = criterion(pred_node.double(), data.mask_node_label[:,0])

        acc_node = self.compute_accuracy(pred_node, data.mask_node_label[:,0])
        # acc_node_accum += acc_node
        acc_edge = 0
        if self.args.mask_edge:
            masked_edge_index = data.edge_index[:, data.connected_edge_indices]
            edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            pred_edge = self.linear_pred_bonds(edge_rep)
            pred_loss += criterion(pred_edge.double(), data.mask_edge_label[:,0])

            acc_edge = self.compute_accuracy(pred_edge, data.mask_edge_label[:,0])
            # acc_edge_accum += acc_edge
        return pred_loss, acc_node, acc_edge


    def get_reference(self, dataloader, number):
            gs = torch.FloatTensor([]).to(self.device)
            for data in dataloader:
                data = data.to(self.device)
                _, g = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
                gs = torch.cat((gs,g),dim=0)
            gs = gs.detach().cpu().numpy()
            n_clusters = number
            kmeans = KMeans(n_clusters=n_clusters, random_state=1)
            kmeans.fit(gs)
            y_pred = kmeans.predict(gs)
            cluster_centers = kmeans.cluster_centers_

            closest_indices = np.zeros(n_clusters, dtype=int)

            # Calculate the distance from each sample to each cluster center
            for i, center in enumerate(cluster_centers):
                distances = np.linalg.norm(gs - center, axis=1)
                closest_indices[i] = np.argmin(distances)

            return closest_indices
    
    def watermark_loss(self, model, key_normal, key_watermark):

        _,normal_h = model(key_normal.x,key_normal.edge_index,key_normal.edge_attr,key_normal.batch)
        _,watermark_h = model(key_watermark.x, key_watermark.edge_index, key_watermark.edge_attr, key_watermark.batch)
        watermark_loss = nn.functional.pairwise_distance(normal_h,watermark_h).mean()
       

        return watermark_loss
    


        

    def CW_IP(self, key_normal, key_watermark, eps, step):
        """
        eps: the ball of finding the worst updates of model parameters
        step: how many steps we apply here
        second_order: whether compute the second order graident
        """

        clone = clone_module(self.encoder)
        
        meta_loss = 0.0
        for _ in range(step):
            wm_loss = self.watermark_loss(clone, key_normal,key_watermark)
            grads = [torch.rand_like(g) for g in clone.parameters()]

            lr = eps/(float(step))

            updates = [lr * g for g in grads] # here apply gradient ascent
            clone = update_module(clone, updates)

            meta_loss = meta_loss + wm_loss

        meta_loss = (meta_loss+self.watermark_loss(clone, key_normal,key_watermark))/(step+1)
        return meta_loss





    def watermarking(self, key_normal, key_watermark, dataloader, args, verbose=True):

        criterion = nn.CrossEntropyLoss()
        #set up optimizers
        # model_list = [self.encoder, self.linear_pred_atoms, self.linear_pred_bonds]

        optimizer_model = torch.optim.Adam(self.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_linear_pred_atoms = torch.optim.Adam(self.linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_linear_pred_bonds = torch.optim.Adam(self.linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]
        self.encoder.train()
        self.linear_pred_atoms.train()
        self.linear_pred_bonds.train()

        for epoch in range(args.num_epochs):
            pred_loss_accum = 0
            acc_node_accum = 0
            acc_edge_accum = 0
            for data in dataloader:
                data = data.to(self.device)
                key_normal = key_normal.to(self.device)
                key_watermark = key_watermark.to(self.device)

                optimizer_model.zero_grad()
                optimizer_linear_pred_atoms.zero_grad()
                optimizer_linear_pred_bonds.zero_grad()

                pred_loss, acc_node, acc_edge = self.pred_loss(data)
                acc_node_accum += acc_node
                acc_edge_accum += acc_edge

                if args.random:
                    wm_loss = self.CW_IP(key_normal, key_watermark, eps=args.eps, step=args.step)
                else:
                    wm_loss = self.watermark_loss(self.encoder, key_normal, key_watermark)
                loss = pred_loss + args.alpha * wm_loss

                loss.backward()

                optimizer_model.step()
                optimizer_linear_pred_atoms.step()
                optimizer_linear_pred_bonds.step()

                pred_loss_accum += float(loss.cpu().item())
            epoch_loss = pred_loss_accum/len(dataloader)
            epoch_acc_node = acc_node_accum/len(dataloader)
            epoch_acc_edge = acc_edge_accum/len(dataloader)
            if verbose and epoch % 5 == 0:
                print('Epoch {}, training loss: {:.6f}, node acc: {:.4f}, edge acc: {:.4f}, Watermark loss: {:.6f}'.format(epoch,epoch_loss,epoch_acc_node,epoch_acc_edge, wm_loss.item()))
        

    def compute_accuracy(self, pred, target):
        return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

    def transfer_finetune(self,train_dataloader, test_dataloader, val_dataloader, args, num_tasks, verbose=True):
        # encoder = copy.deepcopy(self.encoder)
        self.fc = nn.Linear(args.num_hidden, num_tasks).to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_fc = torch.optim.Adam(self.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.train()
        best_val_acc = 0
        for epoch in range(args.num_epochs):

            epoch_loss = 0

            for data in train_dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()
                optimizer_fc.zero_grad()

                cont_loss = self.finetune_BCE_loss(self.encoder, self.fc, data)
                loss = cont_loss 

                loss.backward()
                optimizer.step()
                optimizer_fc.step()

                epoch_loss += cont_loss.item()
            epoch_loss = epoch_loss/len(train_dataloader)

<<<<<<< HEAD
            if (epoch+1) % 5 == 0:
                if(verbose):
                    print('Epoch {}, BCE loss: {:.6f}'\
                        .format(epoch,epoch_loss))
                train_acc = self.transfer_eval(self.encoder, self.fc, train_dataloader)
                val_acc = self.transfer_eval(self.encoder, self.fc, val_dataloader)
                test_acc = self.transfer_eval(self.encoder, self.fc, test_dataloader)
                if(verbose):
                    print('Epoch {}, train roc: {}, val roc: {}, test roc: {}'.format(epoch, train_acc, val_acc, test_acc))
=======
            if verbose and (epoch+1) % 5 == 0:
                print('Epoch {}, BCE loss: {:.6f}'\
                      .format(epoch,epoch_loss))
                train_acc = self.transfer_eval(self.encoder, self.fc, train_dataloader)
                val_acc = self.transfer_eval(self.encoder, self.fc, val_dataloader)
                test_acc = self.transfer_eval(self.encoder, self.fc, test_dataloader)
                
                print('Epoch {}, train roc: {}, val roc: {}, test roc: {}'.format(epoch, train_acc, val_acc, test_acc))
>>>>>>> b99538ad237ca2f0286619ea31de9e82c9204f15
                if(val_acc > best_val_acc):
                    best_val_acc = val_acc
                    weights = deepcopy(self.state_dict())

        print('Epoch {}, BCE loss: {:.6f}'\
                .format(epoch+1,epoch_loss))
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def finetune_BCE_loss(self, model, fc, data):
        criterion = nn.BCEWithLogitsLoss(reduction = "none")
        _, g = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = fc(g)
        # print(pred)
        y = data.y.view(pred.shape).to(torch.float64)
        #Whether y is non-null or not.
        is_valid = y**2 > 0
        loss_mat = criterion(pred.double(), (y+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        return loss
    
    def transfer_eval(self, encoder, fc, test_dataloader):
        encoder.eval()
        y_true = []
        y_scores = []
        for data in test_dataloader:
            data = data.to(self.device)
            with torch.no_grad():
                _, g = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
                pred = fc(g)
            y_true.append(data.y.view(pred.shape))
            y_scores.append(pred)
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

        # print(y_true)
        # print(y_scores)
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
    
    def test(self, train_loader,test_loader, key_normal, key_watermark, args):
        self.eval()
        self.fc.eval()

        # x_test = []
        pred_test = []
        y_test = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                _, g = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
                # x_test.append(g.detach())
                pred = self.fc(g).detach().cpu()
                pred_test.append(pred)
                y_test.append(data.y.detach().cpu().view(pred.shape))
            # x_test = torch.cat(x_test, dim=0)
            pred_test = torch.cat(pred_test, dim = 0)
            y_test = torch.cat(y_test, dim=0)

        # pred_test = torch.softmax(fc(x_test),dim=1)
        accuracy_list = []
        for i in range(y_test.shape[1]):
            acc, _ = self.multi_task_accuracy(pred_test,y_test,i)
            accuracy_list.append(acc)

        if len(accuracy_list) < y_test.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" % (1 - float(len(accuracy_list)) / y_test.shape[1]))
        avg_acc = sum(accuracy_list) / len(accuracy_list)

        if(key_normal == None or key_watermark == None):
            if args.debug:
                print("Accuracy: {:.4f}".format(avg_acc))
            return float(avg_acc)
        else:
            preds_normal = []
            preds_watermark = []
            ys_normal = []
            ys_watermark = []
            key_normal = key_normal.to(self.device)
            key_watermark = key_watermark.to(self.device)
            _,normal_h = self.encoder(key_normal.x,key_normal.edge_index,key_normal.edge_attr,key_normal.batch)
            _,watermark_h = self.encoder(key_watermark.x, key_watermark.edge_index, key_watermark.edge_attr, key_watermark.batch)
            preds_normal.append(self.fc(normal_h).detach().cpu())
            preds_watermark.append(self.fc(watermark_h).detach().cpu())
            preds_normal = torch.cat(preds_normal, dim = 0)
            preds_watermark = torch.cat(preds_watermark, dim=0)
            
            IP_accs = []
            for i in range(y_test.shape[1]):
                logit_normal = preds_normal[:, i]
                prediction_normal = (logit_normal > 0).int()
                logit_watermark = preds_watermark[:, i]
                prediction_watermark = (logit_watermark > 0).int()
                result = torch.concat([prediction_normal,prediction_watermark])
                correct = prediction_normal.eq(prediction_watermark).double()
                IP_acc = correct.sum()/len(prediction_watermark)
                IP_accs.append(IP_acc)
            avg_IP_acc = np.mean(IP_accs)

            if args.debug:
                print("Accuracy: {:.4f}, IP_ACC: {:.4f}".format(avg_acc, avg_IP_acc))

            return float(avg_acc), float(avg_IP_acc)
    
    def multi_task_accuracy(self,pred_test,y_test,i):
        accuracy_list = []
        # for i in range(y_test.shape[1]):
        is_valid = y_test[:, i]**2 > 0
        valid_true = y_test[is_valid, i]
        valid_scores = pred_test[is_valid, i]

        # Convert scores to binary predictions
        predictions = (valid_scores > 0).int()

        # Convert labels to 0 and 1
        valid_true = (valid_true + 1) / 2
        # Calculate accuracy
        accuracy = torch.mean((predictions == valid_true).float()).numpy()
        return accuracy, predictions
    
    def test_finetune(self, train_loader,test_loader, key_normal, key_watermark, num_tasks, args):
        
        refer = self.encoder
        model = copy.deepcopy(self.encoder)
        # num_classes = train_loader.dataset.dataset.data.y.max().item() + 1
        fc = nn.Linear(args.num_hidden, num_tasks).to(self.device)
        optimizer_fc = torch.optim.Adam(fc.parameters(), lr=0.0003, weight_decay=args.weight_decay)
        optimizer_enc = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=args.weight_decay)


        
        for epoch in range(args.test_iteration):
            model.train()
            y_train = []
            for data in train_loader:

                optimizer_fc.zero_grad()
                optimizer_enc.zero_grad()
                data = data.to(self.device)
                _, g = model(data.x, data.edge_index, data.edge_attr, data.batch)
                pred = fc(g)
                y_train.append(data.y.detach().cpu().view(pred.shape))

                # loss = torch.nn.functional.cross_entropy(pred,data.y)
                loss = self.finetune_BCE_loss(model, fc, data)
                loss.backward()
                optimizer_fc.step()
                optimizer_enc.step()
            y_train = torch.cat(y_train, dim=0)

            w_diff = float(parameter_diff(model.parameters(), refer.parameters()).detach())


            preds_normal = []
            preds_watermark = []
            ys_normal = []
            ys_watermark = []
            key_normal = key_normal.to(self.device)
            key_watermark = key_watermark.to(self.device)
            _,normal_h = self.encoder(key_normal.x,key_normal.edge_index,key_normal.edge_attr,key_normal.batch)
            _,watermark_h = self.encoder(key_watermark.x, key_watermark.edge_index, key_watermark.edge_attr, key_watermark.batch)
            preds_normal.append(self.fc(normal_h).detach().cpu())
            preds_watermark.append(self.fc(watermark_h).detach().cpu())
            preds_normal = torch.cat(preds_normal, dim = 0)
            preds_watermark = torch.cat(preds_watermark, dim=0)
            
            IP_accs = []
            for i in range(y_train.shape[1]):
                logit_normal = preds_normal[:, i]
                prediction_normal = (logit_normal > 0).int()
                logit_watermark = preds_watermark[:, i]
                prediction_watermark = (logit_watermark > 0).int()
                result = torch.concat([prediction_normal,prediction_watermark])
                correct = prediction_normal.eq(prediction_watermark).double()
                IP_acc = correct.sum()/len(prediction_watermark)
                IP_accs.append(IP_acc)
            avg_IP_acc = np.mean(IP_accs)


            model.eval()
            pred_test = []
            y_test = []
            with torch.no_grad():
                for data in test_loader:

                    data = data.to(self.device)
                    _, g = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
                    # x_test.append(g.detach())
                    pred = self.fc(g).detach().cpu()
                    pred_test.append(pred)
                    y_test.append(data.y.detach().cpu().view(pred.shape))
                pred_test = torch.cat(pred_test, dim = 0)
                y_test = torch.cat(y_test, dim=0)

            accuracy_list = []
            for i in range(y_test.shape[1]):
                acc, _ = self.multi_task_accuracy(pred_test,y_test,i)
                accuracy_list.append(acc)

            if len(accuracy_list) < y_test.shape[1]:
                print("Some target is missing!")
                print("Missing ratio: %f" % (1 - float(len(accuracy_list)) / y_test.shape[1]))
            avg_acc = sum(accuracy_list) / len(accuracy_list)

            if args.debug:
                print("Epoch: {}, Fine-tune Acc: {:.4f}, IP_ACC: {:.4f}, w_diff: {}"\
                    .format(epoch, avg_acc, avg_IP_acc, w_diff))
                
        return float(avg_acc), float(avg_IP_acc)

