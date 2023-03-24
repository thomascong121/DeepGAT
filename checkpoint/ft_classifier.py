from omegaconf import DictConfig
import pandas as pd
import hydra
import time
import cv2
import torch.nn as nn
from tqdm import tqdm
import copy
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
from util.util import My_Transform, My_Normalize, metric


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class WSIDataset(Dataset):
    """Generate dataset."""
    def __init__(self, filepath):
        self.all_data = pd.read_csv(filepath)
        all_imgpth = self.all_data['image'].tolist()
        all_label = self.all_data['label'].tolist()
        # self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.stage_split(all_imgpth, all_label)
        self.X_train = all_imgpth
        self.y_train = all_label
        print('Train detail: ')
        train_normal, train_tumor = self.Obtain_detail(self.y_train)
        print('normal vs tumor = %d vs %d'%(train_normal, train_tumor))
        self.transform_train = transforms.Compose([My_Transform(), My_Normalize()])
        self.transform_valid = transforms.Compose([My_Normalize(pil=False)])

    def stage_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
        print('Train detail: ')
        train_normal, train_tumor = self.Obtain_detail(y_train)
        print('normal vs tumor = %d vs %d'%(train_normal, train_tumor))
        print('Valid detail: ')
        valid_normal, valid_tumor = self.Obtain_detail(y_val)
        print('normal vs tumor = %d vs %d'%(valid_normal, valid_tumor))
        print('Test detail: ')
        test_normal, test_tumor = self.Obtain_detail(y_test)
        print('normal vs tumor = %d vs %d'%(test_normal, test_tumor))
        return X_train, X_val, X_test, y_train, y_val, y_test

    def Obtain_detail(self, y):
        normal = tumor = 0
        for i in y:
            if i == 'Normal':
                normal += 1
            else:
                tumor += 1
        return normal, tumor

    def Obtain_dataset(self, stage):
        if stage == 'Train':
            self.dataset = StageDataset(self.X_train, self.y_train, self.transform_train)
        elif stage == 'Valid':
            self.dataset = StageDataset(self.X_val, self.y_val, self.transform_valid)
        elif stage == 'Test':
            self.dataset = StageDataset(self.X_test, self.y_test, self.transform_valid)
        return self.dataset

    def Obtain_loader(self, stage, batch_size):
        ds = self.Obtain_dataset(stage)
        self.loader = DataLoader(ds,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 drop_last=True)
        return self.loader


class StageDataset(Dataset):
    def __init__(self, X, y, transform):
        self.map_dict = {'Normal': 0, 'Tumor': 1}
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        '''image	label'''
        img_path = self.X[idx]
        label = self.map_dict[self.y[idx]]
        # image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

        sample = {'image': img_path, 'label': label, 'pth':img_path}

        if self.transform:
            sample = self.transform(sample)

        return sample

@hydra.main(config_path='configs', config_name='config.yaml')
def runner(cfg: DictConfig):
    data_root = cfg.Training.train_class_csv
    batch_size = cfg.Training.batch_size
    num_epochs = cfg.Training.n_epoch
    model_save_path = os.path.join(cfg.Training.checkpoint_dir, 'Pretrained_ReNet50_torch.pt')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('\n Using : ', device)

    wsi_dataset = WSIDataset(data_root)
    trainloader = wsi_dataset.Obtain_loader('Train', batch_size)
    # validloader = wsi_dataset.Obtain_loader('Valid', batch_size)
    # testloader = wsi_dataset.Obtain_loader('Test', batch_size)

    # check readed data
    print('number of training data %d' % len(trainloader))
    # print('number of validation data %d' % len(validloader))
    # print('number of testing data %d' % (len(testloader)))

    # define model
    model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    set_parameter_requires_grad(model, False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)

    since = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)  # , weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        dataloaders = {'train': trainloader}#, 'val': validloader}
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train']:#, 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                inputs = data['image'].to(device=device, dtype=torch.float)
                labels = data['label'].to(device=device, dtype=torch.int64)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    labels = labels.squeeze()
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
            epoch_acc = metric(dataloaders[phase], model)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f} Lr: {}'.format(phase, epoch_loss, epoch_acc,
                                                              optimizer.param_groups[0]["lr"]))
            print('========================')
            # deep copy the modela
            if phase == 'train':
                val_loss = epoch_loss
                best_acc = epoch_acc
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, model_save_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # model.load_state_dict(best_model_wts)
    # metric(testloader, model)

if __name__ == '__main__':
    runner()