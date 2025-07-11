import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import accuracy
import copy
from utils import parameter_diff, clone_module, update_module


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, project, contrast_model, device):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.project = project
        self.contrast_model = contrast_model
        self.device = device

       

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
        watermark_loss = nn.functional.pairwise_distance(self.embedding,watermark_h).mean()
       
        return watermark_loss
    
    def meta_IP(self, key_watermark, eps, step, random):
        """
        eps: the ball of finding the worst updates of model parameters
        step: how many steps we apply here
        second_order: whether compute the second order graident
        """

        clone = clone_module(self.encoder)
        
        meta_loss = 0.0
        for _ in range(step):
            wm_loss = self.watermark_loss(clone,key_watermark)

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

        meta_loss = (meta_loss+self.watermark_loss(clone, key_watermark))/(step+1)
        return meta_loss


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
                wm_loss = self.watermark_loss(self.encoder,key_watermark)
                meta_loss = self.meta_IP(key_watermark, eps=args.eps, step=args.step, random=args.random)
                loss = cont_loss + args.alpha * meta_loss

                loss.backward()
                optimizer.step()

                epoch_loss += cont_loss.item()
            epoch_loss = epoch_loss/len(dataloader)
            # print(self.embedding[:10])
            if verbose and (epoch+1) % 5 == 0:
                print('Epoch {}, contrastive loss: {:.6f}, Watermark loss: {:.6f}, Meta IP loss: {:.6f}'\
                        .format(epoch+1,epoch_loss, wm_loss.item(), meta_loss.item()))
        print('Epoch {}, contrastive loss: {:.6f}, Watermark loss: {:.6f}, Meta IP loss: {:.6f}'\
        .format(epoch+1,epoch_loss, wm_loss.item(), meta_loss.item()))

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
        _,watermark_h = self.encoder(key_watermark.x, key_watermark.edge_index, key_watermark.batch)

        score_watermark, pred_watermark = torch.softmax(fc(watermark_h.detach()),dim=1).max(1)

        # print(pred_watermark)
        pred_watermark = torch.eye(num_classes,device=self.device).float()[pred_watermark]
        IP_acc = (pred_watermark.float().mean(dim=0)).max().item()

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
            score_watermark, pred_watermark = torch.softmax(fc(watermark_h.detach()),dim=1).max(1)

            pred_watermark = torch.eye(num_classes,device=self.device).float()[pred_watermark]
            IP_acc = (pred_watermark.float().mean(dim=0)).max().item()



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

