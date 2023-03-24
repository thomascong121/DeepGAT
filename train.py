import hydra
import torch
import numpy as np
import os
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch_geometric.transforms as T
from network.gcn import GCN
from util.read_data import WSIDataset
from omegaconf import DictConfig
from util.util import get_idx, obtain_identity_adj, accuracy, adaptation_factor

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

@hydra.main(config_path='./configs', config_name='config.yaml')
def runner(cfg: DictConfig):
    feature_size = cfg.Training.feature_size
    hidden_size = cfg.Training.hidden_size
    times = cfg.Training.times
    n_class = cfg.Training.n_class
    epoch = cfg.Training.n_epoch
    dropout = cfg.Training.dropout
    train_pth = cfg.Training.train_pth
    lr = cfg.Training.lr
    weight_decay = cfg.Training.weight_decay
    number_labelled = cfg.Training.number_labelled
    checkpoint_dir = cfg.Training.checkpoint_dir

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Constructing Dataset....')
    train_dataset = WSIDataset(train_pth, checkpoint_dir)#, transform=T.NormalizeFeatures())

    print('Start Training....')
    tumor_acc = []
    for i in tqdm(range(len(train_dataset))):
        # loop over all graphs
        test_acc = []
        use_index = []
        data = train_dataset[i]
        print('data ',data)
        for t in range(times):
            features = data.x.cuda()
            labels = data.y.long().cuda()
            adj = data.edge_index.cuda()
            adj_cnn = obtain_identity_adj(data)
            idx_train, idx_test, neg2pos = get_idx(data, number_labelled)
            # print('Train vs Test ',len(idx_train),len(idx_test))
            model = GCN(feature_size, hidden_size, n_class, dropout)
            model = model.to(device)
            params = list(model.parameters())
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
            best_acc = 0
            best_epoch = 0
            # idx_train = torch.tensor([334, 267])

            for ep in range(epoch):
                model.train()
                optimizer.zero_grad()
                output, feature, cnn_output, cnn_feature, gcn_features1, cnn_features1 = model(features, adj, adj_cnn.cuda())
                # lam = 10#adaptation_factor(epoch / 50)
                weight = [1, 1]
                loss_train_H = model.loss(output[idx_train], labels[idx_train].long(), weight)
                loss_train_F = model.loss(cnn_output[idx_train],labels[idx_train].long(), weight)
                semantic_loss_1 = model.adloss1(cnn_features1[idx_train], gcn_features1, labels[idx_train], output)
                semantic_loss_2 = model.adloss(cnn_feature[idx_train], feature, labels[idx_train], output)

                mu1 = 1
                mu2 = 0.1
                lam1 = 0
                lam2 = 0
                loss_train = mu1 * loss_train_H + mu2 * loss_train_F + lam1 * semantic_loss_1 + lam2 * semantic_loss_2
                acc_train = accuracy(output[idx_train], labels[idx_train])#, detail=True)

                loss_train.backward()
                optimizer.step()
                acc_test = accuracy(output[idx_test], labels[idx_test])#, detail=True)

                print('Epoch: {:04d}'.format(ep),
                        'loss_train_H: {:.4f}'.format(loss_train_H.item()),
                        'loss_train_F: {:.4f}'.format(loss_train_F.item()),
                        'acc_train: {:.4f}'.format(acc_train.item()),
                        'acc_test: {:.4f}'.format(acc_test.item()),
                        'semantic_loss_layer1: {:.4f}'.format(semantic_loss_1.item()),
                        'semantic_loss_layer2: {:.4f}'.format(semantic_loss_2.item())
                      )

                if acc_test >= best_acc:
                    best_acc = acc_test
                    best_epoch = ep
                    torch.save(model.state_dict(), checkpoint_dir+'/{}.pkl'.format(ep))

                for f in os.listdir(checkpoint_dir):
                    if f.endswith('.pkl'):
                        epoch_nb = int(f.split('.')[0])
                        if epoch_nb < best_epoch:
                            os.remove(checkpoint_dir+'/%s' % f)

            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pkl'):
                    epoch_nb = int(f.split('.')[0])
                    if epoch_nb > best_epoch:
                        os.remove(checkpoint_dir+'/%s' % f)
            # print('Best epoch is ',best_epoch)
            model.load_state_dict(torch.load(checkpoint_dir+'/{}.pkl'.format(best_epoch)))
            model.eval()
            output, feature, cnn_output, cnn_feature, gcn_features1, cnn_features1 = model(features, adj, adj_cnn.cuda())
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test], detail=True)
            # print("Test set results:",
            #       "loss= {:.4f}".format(loss_test.item()),
            #       "accuracy= {:.4f}".format(acc_test.item()))
            print(idx_train)
            use_index.append(idx_train)
            test_acc.append(acc_test.item())
        print('acc_test: {:.4f}'.format(np.mean(test_acc)))
        tumor_acc.append(np.mean(test_acc))
        break

    print('overall acc_test: {:.4f}'.format(np.mean(tumor_acc)))





'''


1 label per class
overall acc_test: 0.6704
5 label per class
overall acc_test: 0.7011
10 label per class
overall acc_test: 0.7323
===new===
1,0,0,0 - GAT
preds pos 10
preds neg 324
number pos  8
number neg  326
number of correct positive 4
number of correct negative 320

1,0,0,0 - GATv2
preds pos 6
preds neg 328
number pos  8
number neg  326
number of correct positive 3
number of correct negative 323

1,0,0,0 - TransformerConv
preds pos 4
preds neg 330
number pos  8
number neg  326
number of correct positive 3
number of correct negative 325

1,0,0,0 - TAGConv
preds pos 11
preds neg 323
number pos  8
number neg  326
number of correct positive 4
number of correct negative 319

1,0,0,0 - SGConv
preds pos 5
preds neg 329
number pos  8
number neg  326
number of correct positive 3
number of correct negative 324

1,0,0,0 - HypergraphConv
preds pos 20
preds neg 314
number pos  8
number neg  326
number of correct positive 8
number of correct negative 314

1,0,0,0 - ClusterGCNConv
preds pos 2
preds neg 332
number pos  8
number neg  326
number of correct positive 2
number of correct negative 326

1,0,0,0 - GENConv
preds pos 29
preds neg 305
number pos  8
number neg  326
number of correct positive 7
number of correct negative 304

1,0,0,0 - FiLMConv
preds pos 7
preds neg 327
number pos  8
number neg  326
number of correct positive 3
number of correct negative 322

1,0,0,0 - SuperGATConv
preds pos 9
preds neg 325
number pos  8
number neg  326
number of correct positive 3
number of correct negative 320

1,1,0,0
preds pos 6
preds neg 328
number pos  8
number neg  326
number of correct positive 3
number of correct negative 323

1,1,1,0
preds pos 6
preds neg 328
number pos  8
number neg  326
number of correct positive 3
number of correct negative 323

1,1,1,1
preds pos 6
preds neg 328
number pos  8
number neg  326
number of correct positive 3
number of correct negative 323

0,1,1,1
preds pos 0
preds neg 334
number pos  8
number neg  326
number of correct positive 0
number of correct negative 326
===old===
1,0,0,0
preds pos 1
preds neg 333
number pos  8
number neg  326

0,1,0,0
preds pos 27
preds neg 307
number pos  8
number neg  326
number of correct  299 334

0,0,1,0
preds pos 0
preds neg 334
number pos  8
number neg  326
number of correct  326.0 334

0,0,0,1
preds pos 0
preds neg 334
number pos  8
number neg  326
number of correct  326.0 334
'''

if __name__ == '__main__':
    runner()