#%%
import torch

#%%
import numpy as np
from torch_geometric.utils import erdos_renyi_graph

class WatermarkGraph:
    def __init__(self,args, device):
        self.args = args
        self.device = device

    def gene_random_graph(self, features, num_nodes, p = 0.5, feat_generation = 'Rand_Gene'):
        
        # generate watermark graph with random edge
        watermark_edge_index = erdos_renyi_graph(num_nodes, p,directed=False)
        watermark_edge_weight = torch.ones(watermark_edge_index.shape[1],dtype=torch.float32)
        # generate node features
        if feat_generation == 'Rand_Gene':
            # print("Rand generate the watermark features")
            features = features.cpu().numpy()
            mean = features.mean(axis=0)
            std = features.std(axis=0)

            watermark_feat = []
            for i in range(num_nodes):
                watermark_feat.append(torch.tensor(np.random.normal(mean,std),dtype=torch.float32))
            watermark_feat = torch.stack(watermark_feat)
        else:
            # print("Rand sample the watermark features from the original features")
            idx = np.random.randint(features.shape[0],size=num_nodes)
            watermark_feat = features[idx]
        watermark_feat = watermark_feat
        return watermark_feat, watermark_edge_index, watermark_edge_weight
    

    def WMRandom(self, features, num_nodes, p = 0.5):

        # generate watermark graph with random edge
        watermark_edge_index = erdos_renyi_graph(num_nodes, p, directed=False)
        watermark_edge_weight = torch.ones(watermark_edge_index.shape[1],dtype=torch.float32)
        # generate node features

        watermark_feat = torch.tensor(np.random.binomial(1,0.2,size=(num_nodes,features.shape[1])),dtype=torch.float32)
        return watermark_feat, watermark_edge_index, watermark_edge_weight
    