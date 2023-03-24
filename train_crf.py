import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm
from data.wsi_dataset import WSIDataset
from network.ResCRF import resnet50, resnet34
import copy


def metric(output, labels):
    TP = TN = FP = FN = 0
    for i in range(output.size()[0]):
        probs = output[i].sigmoid()
        predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)
        for j in range(len(predicts)):
            patch_pred = predicts[j].cpu().item()
            patch_label = labels[i][j].cpu().item()
            if patch_pred == patch_label == 0:
                TN += 1
            elif patch_pred == patch_label == 1:
                TP += 1
            elif patch_pred == 0 and patch_label == 1:
                FN += 1
            else:
                FP += 1
    return TN, TP, FN, FP

num_epochs = 60
batch_size = 40
data_pth = '/home/congz3414050/HistoGCN/data/5X/Tumor_768/all_data.csv'
model_save_path = '/home/congz3414050/HistoGCN/checkpoint/Scratch_Res34CRF_torch.pt'
annotation_path = '/home/congz3414050/HistoGCN/data/Original/annotation'
wsi_dataset = WSIDataset(data_pth, crf=True, annotation_pth=annotation_path)
trainloader = wsi_dataset.Obtain_loader('Train', batch_size)
testloader = wsi_dataset.Obtain_loader('Test', batch_size)

# check readed data
print('number of training data %d' % len(trainloader))
print('number of testing data %d' % (len(testloader)))
model = resnet34(num_classes=1, num_nodes=9)
# model.load_state_dict(torch.load(model_save_path))

print('loading from checkpoint')
since = time.time()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)  # , weight_decay=0.01)
criterion = nn.BCEWithLogitsLoss()
best_acc = 0.0
best_loss = float('inf')
best_f1 = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    dataloaders = {'train': trainloader, 'val': testloader}
    batch_TP = 1
    batch_TN = 1
    batch_FP = 1
    batch_FN = 1
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    stop_count = 0
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode
        running_loss = 0.0

        # Iterate over data.
        for data in tqdm(dataloaders[phase]):
            (img_flat, label_flat) = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                pred = model(img_flat)
                pred = pred.squeeze(-1)

                loss = criterion(pred, label_flat)
                TN, TP, FN, FP = metric(pred, label_flat)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * img_flat.size(0)
            batch_TN += TN
            batch_TP += TP
            batch_FN += FN
            batch_FP += FP
            stop_count += 1
        #             if stop_count > 10:
        #                 break
        #             break
        # epoch_f1 = metric(dataloaders[phase], graphloader, model, batch_size, use_graph=use_graph)
        print(batch_TN, batch_TP, batch_FN, batch_FP)
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        Specificity = batch_TN / (batch_TN + batch_FP)
        Sensitivity = batch_TP / (batch_FN + batch_TP)
        Precision = batch_TP / (batch_TP + batch_FP)
        F1_Score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)

        print('Stage %s' % phase)
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
            print('===========End of Epoch %d=============' % epoch)
        # break
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val f1: {:4f} at epoch {:d}'.format(best_f1, best_epoch))