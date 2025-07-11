import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
import copy
from utils import parameter_diff
from torch_geometric.data import Data


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class Encoder(torch.nn.Module):
    def __init__(self, encoder, device):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.device = device


    def forward(self, x, edge_index, batch):

        z1, g1 = self.encoder(x, edge_index, batch)
        return z1, g1


    def watermarking(self, teacher, dataloader, args, verbose=True):

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.train()
        for epoch in range(args.num_epochs):

            epoch_loss = 0

            for data in dataloader:
                data = data.to(self.device)

                optimizer.zero_grad()

                _, teacher_h = teacher.encoder(data.x_s,data.edge_index_s,data.x_s_batch)
                _, student_h = self.encoder(data.x_s,data.edge_index_s,data.x_s_batch)
                utility_loss = 1 - F.cosine_similarity(teacher_h,student_h).mean()

                _, wm_h = self.encoder(data.x_t,data.edge_index_t, data.x_t_batch)
                wm_loss = F.cosine_similarity(teacher_h, wm_h).mean()
                loss = utility_loss + args.alpha * wm_loss

                loss.backward()
                optimizer.step()

                epoch_loss += utility_loss.item()
            epoch_loss = epoch_loss/len(dataloader)
            # print(self.embedding[:10])
            if verbose and (epoch+1) % 5 == 0:
                print('Epoch {}, utility loss: {:.6f}, Watermark loss: {:.6f}'\
                      .format(epoch+1,epoch_loss, wm_loss.item()))
        

    def test(self, train_loader,test_loader, key_watermark, args):
        self.eval()
        x_train = []
        y_train = []

        with torch.no_grad():
            for data in train_loader:
                data = data.to(self.device)
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
        
        key_watermark = key_watermark.to(self.device)

        _,clean_h = self.encoder(key_watermark.x_s, key_watermark.edge_index_s, key_watermark.x_s_batch)
        _, pred_clean = torch.softmax(fc(clean_h.detach()),dim=1).max(1)
        _,watermark_h = self.encoder(key_watermark.x_t, key_watermark.edge_index_t, key_watermark.x_t_batch)
        _, pred_watermark = torch.softmax(fc(watermark_h.detach()),dim=1).max(1)

        # print(pred_watermark)
        
        IP_acc = 1-(pred_watermark==pred_clean).float().mean().item()

        # print("Accuracy: {:.4f}, IP_ACC: {:.4f}".format(acc, IP_acc))
        return float(acc), float(IP_acc)
    

    def test_finetune(self, train_loader,test_loader, key_watermark, args):
        
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

            key_watermark = key_watermark.to(self.device)
            _,clean_h = self.encoder(key_watermark.x_s, key_watermark.edge_index_s, key_watermark.x_s_batch)
            _, pred_clean = torch.softmax(fc(clean_h.detach()),dim=1).max(1)
            _, watermark_h = self.encoder(key_watermark.x_t, key_watermark.edge_index_t, key_watermark.x_t_batch)
            _, pred_watermark = torch.softmax(fc(watermark_h.detach()),dim=1).max(1)

            # print(pred_watermark)
            
            IP_acc = 1-(pred_watermark==pred_clean).float().mean().item()

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

        return float(acc), float(IP_acc)

