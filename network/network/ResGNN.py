import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv, \
    TAGConv, SGConv, HypergraphConv, ClusterGCNConv, GENConv, FiLMConv, SuperGATConv, \
    HGTConv

class ResGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGNN, self).__init__()
        base_model = torchvision.models.resnet50()
        self.feature_et = nn.Sequential(*list(base_model.children())[:-2])
        self.gcn1 = GATConv(nfeat, nhid)
        self.gcn2 = GATConv(nhid, nclass)
        self.dropout = dropout
        self.MSEloss =  nn.MSELoss()

    def forward(self, x, graph_data):
        x = self.feature_et(x)
        x = x.view(x.shape[0] * x.shape[2] * x.shape[3], x.shape[1])
        adj = graph_data.edge_index

        # gcn features
        x_gcn = F.relu(self.gcn1(x, adj))
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        x_gcn = self.gcn2(x_gcn, adj)
        gcn_features = x_gcn

        return gcn_features

    def criterion(self, input, label, bn):
        n_repeat = int(input.size()[0]/bn)
        # print('input/label size ', input.size(), label.size())
        # torch.tensor([n_repeat, n_repeat]).cuda()
        label = torch.repeat_interleave(label, repeats=n_repeat, dim=0).float()
        # print('input/label size ', input.size(), label.size(), label)
        return torch.nn.BCEWithLogitsLoss()(input, label)


if __name__ == '__main__':
    from util.util import metric, get_adj
    import numpy as np
    from torch_geometric.data import Data, DataLoader

    adj_matrx = get_adj(8)
    adj_matrx = np.array(adj_matrx)
    adj_matrx = torch.tensor(adj_matrx, dtype=torch.long)
    adj_matrx = adj_matrx.t().contiguous()
    data_list = []
    for i in range(3):
        data_list.append(Data(x=torch.rand(1, 1), edge_index=adj_matrx))
    graphloader = DataLoader(data_list, batch_size=3)
    my_model = ResGNN(2048, 1024, 1, 0.5)
    x = torch.rand(3,3,256,256)
    label = torch.ones((3))
    for d in graphloader:
        print(d)
        out = my_model(x, d)
        print('out feature ', out.size())
        loss = my_model.criterion(out_ft, label, 3)
        print('loss is ', loss)