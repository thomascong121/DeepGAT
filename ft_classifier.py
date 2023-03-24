from omegaconf import DictConfig
import hydra
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import copy
import os
import torch
from util.util import metric, get_adj
from data.wsi_dataset import WSIDataset
import torchvision
from network.ResGNN import ResGNN
from network.DenseGNN import DenseGNN
from torch_geometric.data import Data, DataLoader


seed = 42
# np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

@hydra.main(config_path='configs', config_name='config.yaml')
def runner(cfg: DictConfig):
    feature1 = cfg.Training.feature_size
    feature2 = cfg.Training.hidden_size
    data_root = cfg.Training.train_class_csv
    batch_size = cfg.Training.batch_size
    num_epochs = cfg.Training.n_epoch
    use_graph = cfg.Training.use_graph
    use_pool = cfg.Training.use_pool
    model_save_path = os.path.join(cfg.Training.checkpoint_dir, 'Scratch_ReNet50SMOTE_torch.pt')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('\n Using : ', device)

    wsi_dataset = WSIDataset(data_root)
    trainloader = wsi_dataset.Obtain_loader('Train', batch_size)
    testloader = wsi_dataset.Obtain_loader('Test', batch_size)

    # check readed data
    print('number of training data %d' % len(trainloader))
    print('number of testing data %d' % (len(testloader)))

    adj_matrx = get_adj(8)
    adj_matrx = np.array(adj_matrx)
    adj_matrx = torch.tensor(adj_matrx, dtype=torch.long)
    adj_matrx = adj_matrx.t().contiguous()
    data_list = []
    for i in range(batch_size):
        data_list.append(Data(x=torch.rand(1, 1), edge_index=adj_matrx))
    graphloader = DataLoader(data_list, batch_size=batch_size)
    batch_idx = torch.tensor([i for i in range(batch_size)])
    batch_idx = batch_idx.unsqueeze(-1)
    batch_idx = torch.repeat_interleave(batch_idx, repeats=64, dim=0).float()
    batch_idx = batch_idx.squeeze(-1)
    batch_idx = batch_idx.long().cuda()

    # define model
    if use_graph:
        print('using cnn+gcn')
        if use_pool:
            model = ResGNN(feature1, feature2, nclass=2, dropout=0.5, batch_idx=batch_idx)
        else:
            model = ResGNN(feature1, feature2, nclass=2, dropout=0.5)
        model.to(device)
    else:
        print('using cnn')
    # model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        model = torchvision.models.resnet50()
        set_parameter_requires_grad(model, False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

        # model.classifier = nn.Linear(1024, 2)
        model.to(device)


    since = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)  # , weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0.0
    best_loss = float('inf')
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        dataloaders = {'train': trainloader, 'val': testloader}
        batch_TP = 0
        batch_TN = 0
        batch_FP = 0
        batch_FN = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                inputs = data['image'].to(device=device, dtype=torch.float)
                labels = data['label'].to(device=device, dtype=torch.int64)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if use_graph:
                        for graph_data in graphloader:
                            outputs = model(inputs, graph_data.cuda())
                            labels = labels.squeeze()
                            loss = model.criterion(outputs, labels, batch_size)
                            TN, TP, FN, FP = metric(outputs, labels, batch_size, use_graph=use_graph, use_pool= not use_pool)
                    else:
                        outputs = model(inputs)
                        labels = labels.squeeze()
                        loss = criterion(outputs, labels.float())
                        # wt = torch.FloatTensor([0.8, 8.0]).cuda()
                        # loss = (loss * wt).mean()
                        TN, TP, FN, FP = metric(outputs, labels, batch_size, use_graph=use_graph, use_pool= use_pool)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                batch_TN += TN
                batch_TP += TP
                batch_FN += FN
                batch_FP += FP

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            Specificity = batch_TN / (batch_TN + batch_FP)
            Sensitivity = batch_TP / (batch_FN + batch_TP)
            Precision = batch_TP / (batch_TP + batch_FP)
            F1_Score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)

            print('Stage %s'%phase)
            print('Specificity: ', Specificity)
            print('Sensitivity: ', Sensitivity)
            print('Precision: ', Precision)
            print('F1-Score: ', F1_Score)
            print('Loss: {:.4f} Lr: {}'.format(epoch_loss, optimizer.param_groups[0]["lr"]))

            # deep copy the modela
            if phase == 'val':
                val_loss = epoch_loss
                if F1_Score > best_f1:
                    best_f1 = F1_Score
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, model_save_path)
                print('===========End of Epoch %d============='%epoch)
            # break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val f1: {:4f} at epoch {:d}'.format(best_f1, best_epoch))
    # model.load_state_dict(best_model_wts)
    # test_f1 = metric(testloader, graphloader, model, batch_size, use_graph=use_graph)
    # metric(testloader, model)

if __name__ == '__main__':
    runner()

'''
patch-level Statistic: 
Specificity:  0.9879154078549849
Sensitivity:  0.765090909090909
Precision:  0.8825503355704698
Acc:  0.9642746913580247
F1-Score:  0.8196338137904168
AUC  0.9759640287205242


patch-level Statistic: 
Specificity:  0.9818592136099297
Sensitivity:  0.8207088255733148
Precision:  0.8496402877697842
Acc:  0.963966049382716
F1-Score:  0.8349240014139272
'''