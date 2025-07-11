import torch
import torch.nn as nn
from utils import accuracy
import copy
from utils import parameter_diff

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, project, contrast_model, watermark, device):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.project = project
        self.contrast_model = contrast_model
        self.device = device
        self.trojan_feat, self.trojan_edge_index = watermark.x, watermark.edge_index
       

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)

        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z1, z2, g1, g2

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


    
    def watermark_loss(self, model, key_watermark):
        _,watermark_h = model(key_watermark.x, key_watermark.edge_index, key_watermark.batch)
        watermark_loss = torch.cdist(watermark_h,watermark_h).mean()
       
        return watermark_loss
    


    def watermarking(self, key_watermark, dataloader, args, verbose=True):

        self.embedding = torch.randn([args.num_hidden],requires_grad=True,device=self.device)

        optimizer = torch.optim.Adam(list(self.parameters()) + [self.embedding], lr=args.lr, weight_decay=args.weight_decay)

        self.train()
        for epoch in range(args.num_epochs):

            epoch_loss = 0

            for data in dataloader:
                data = data.to(self.device)
                key_watermark = key_watermark.to(self.device)

                optimizer.zero_grad()

                cont_loss = self.contrastive_loss(data)
                wm_loss = self.watermark_loss(self.encoder, key_watermark)
                loss = cont_loss + args.alpha * wm_loss

                loss.backward()
                optimizer.step()

                epoch_loss += cont_loss.item()
            epoch_loss = epoch_loss/len(dataloader)
            # print(self.embedding[:10])
            if verbose and (epoch+1) % 5 == 0:
                print('Epoch {}, contrastive loss: {:.6f}, Watermark loss: {:.6f}'\
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
        entropy_test =  -torch.sum(pred_test * torch.log(pred_test.clamp(min=1e-9)), dim=1)
       
        acc = accuracy(pred_test,y_test)
        
        key_watermark = key_watermark.to(self.device)
        _,watermark_h = self.encoder(key_watermark.x, key_watermark.edge_index, key_watermark.batch)

        pred_watermark = torch.softmax(fc(watermark_h.detach()),dim=1)

        # print(pred_watermark)
        entropy_watermark = -torch.sum(pred_watermark * torch.log(pred_watermark.clamp(min=1e-9)), dim=1)

        entropy_all = torch.cat([entropy_test,entropy_watermark])
        median = torch.median(entropy_all)
        mad = torch.median(torch.abs(entropy_all-median))
        outlier_index = torch.abs(median - entropy_watermark)/ (1.4826 * mad)
        IP_acc = (outlier_index > 3.0).float().mean()

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
            _,watermark_h = self.encoder(key_watermark.x, key_watermark.edge_index, key_watermark.batch)
            pred_watermark = torch.softmax(fc(watermark_h.detach()),dim=1)
            entropy_watermark = -torch.sum(pred_watermark * torch.log(pred_watermark.clamp(min=1e-9)), dim=1)


            model.eval()
            pred = []
            y = []
            with torch.no_grad():
                for data in test_loader:

                    data = data.to(self.device)
                    _, g = model(data.x, data.edge_index, data.batch)

                    pred.append(torch.softmax(fc(g),dim=1))
                    y.append(data.y)

            pred = torch.cat(pred, dim=0)
            y = torch.cat(y, dim=0)
            acc = accuracy(pred,y)
            entropy_test =  -torch.sum(pred * torch.log(pred.clamp(min=1e-9)), dim=1)

            entropy_all = torch.cat([entropy_test,entropy_watermark])
            median = torch.median(entropy_all)
            mad = torch.median(torch.abs(entropy_all-median))
            outlier_index = torch.abs(median - entropy_watermark)/ (1.4826 * mad)
            IP_acc = (outlier_index > 3.0).float().mean()

        if args.debug:
            print("Epoch: {}, Fine-tune Acc: {:.4f}, IP_ACC: {:.4f}, w_diff: {}"\
                .format(epoch, acc, IP_acc, w_diff))

        return float(acc), float(IP_acc)

