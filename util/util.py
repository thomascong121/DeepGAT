import numpy as np
import torch.nn as nn
import torch
import math
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from PIL import Image
import torchvision.transforms as transforms


def get_adj(size):
    index_matrix = [[j*size + i for i in range(size)] for j in range(size)]
    adj = []
    for x in range(0, size):
        for y in range(0, size):
            # upper row
            if x-1 >= 0:
                if y-1 >= 0:
                    adj.append([index_matrix[x][y], index_matrix[x-1][y-1]])
                adj.append([index_matrix[x][y], index_matrix[x-1][y]])
                if y+1 < size:
                    adj.append([index_matrix[x][y], index_matrix[x-1][y+1]])
            # current row
            if y-1 >= 0:
                adj.append([index_matrix[x][y], index_matrix[x][y-1]])
            # adj.append([x,y])
            if y+1 < size:
                adj.append([index_matrix[x][y], index_matrix[x][y+1]])
            # lower row
            if x+1 < size:
                if y - 1 >= 0:
                    adj.append([index_matrix[x][y], index_matrix[x + 1][y - 1]])
                adj.append([index_matrix[x][y], index_matrix[x + 1][y]])
                if y + 1 < size:
                    adj.append([index_matrix[x][y], index_matrix[x + 1][y + 1]])
        # adj.sort(key = lambda x:x[0])
    return adj

def get_similarity(size, feature_matrix):
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    wt = []
    for x in range(0, size):
        for y in range(0, size):
            # upper row
            if x-1 >= 0:
                if y-1 >= 0:
                    wt.append(cos_sim(feature_matrix[:,x,y,:], feature_matrix[:,x-1,y-1,:]).tolist())
                wt.append(cos_sim(feature_matrix[:, x, y, :], feature_matrix[:, x - 1, y, :]).tolist())
                if y+1 < size:
                    wt.append(cos_sim(feature_matrix[:, x, y, :], feature_matrix[:, x - 1, y + 1, :]).tolist())
            # current row
            if y-1 >= 0:
                wt.append(cos_sim(feature_matrix[:, x, y, :], feature_matrix[:, x, y - 1, :]).tolist())
            # adj.append([x,y])
            if y+1 < size:
                wt.append(cos_sim(feature_matrix[:, x, y, :], feature_matrix[:, x, y + 1, :]).tolist())
            # lower row
            if x+1 < size:
                if y - 1 >= 0:
                    wt.append(cos_sim(feature_matrix[:, x, y, :], feature_matrix[:, x+1, y - 1, :]).tolist())
                wt.append(cos_sim(feature_matrix[:, x, y, :], feature_matrix[:, x + 1, y, :]).tolist())
                if y + 1 < size:
                    wt.append(cos_sim(feature_matrix[:, x, y, :], feature_matrix[:, x + 1, y + 1, :]).tolist())
    wt = torch.tensor(wt) # number_edge, batch
    n_e, n_b = wt.size()
    wt = wt.view(n_b, n_e) # batch, number_edge
    return wt

def get_idx(data, num_labelled):
    labels = data.y

    # obtain train/val/test idx - ignore val for now
    label_pos = (labels == 1).nonzero(as_tuple=True)[0]
    label_neg = (labels == 0).nonzero(as_tuple=True)[0]
    print('total positive ',len(label_pos))
    print('total negative ', len(label_neg))
    neg2pos = len(label_neg)/len(label_pos)
    # select train idx
    index_pos_train = label_pos.float().multinomial(num_samples=num_labelled, replacement=True)
    index_neg_train = label_neg.float().multinomial(num_samples=num_labelled, replacement=True)
    index_train = torch.cat((label_pos[index_pos_train].long(), label_neg[index_neg_train].long()), 0)
    # use the rest as test idx
    label_pos_test = []
    label_neg_test = []
    for i in label_pos:
        if i not in label_pos[index_pos_train]:
            label_pos_test.append(i)
    for i in label_neg:
        if i not in label_neg[index_neg_train]:
            label_neg_test.append(i)

    index_test = torch.cat((torch.tensor(label_pos_test).long(), torch.tensor(label_neg_test).long()), 0)
    return index_train, index_test, neg2pos

def obtain_identity_adj(data):
    n = data.size()[0]
    source, target = np.arange(n).astype(int), np.arange(n).astype(int)
    adj_edge_list = np.concatenate(([source], [target]), axis=0)
    adj_edge_list = torch.tensor(adj_edge_list, dtype=torch.long).contiguous()
    return adj_edge_list

def accuracy(output, labels, detail=False):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    if detail:
        print('preds pos', len(preds[preds == 1]))
        print('preds neg', len(preds[preds == 0]))
        print('number pos ', len(labels[labels == 1]))
        print('number neg ', len(labels[labels == 0]))
        correct_pos = 0
        correct_neg = 0
        for i in range(len(labels)):
            if labels[i] == preds[i] == 0:
                correct_neg += 1
            elif labels[i] == preds[i] == 1:
                correct_pos += 1
            else:
                continue
        print('number of correct positive', correct_pos)
        print('number of correct negative', correct_neg)
        print()
    return correct / len(labels)

def adaptation_factor(x):
    den = 1.0 + math.exp(-10 * x)
    lamb = 2.0 / den - 1.0
    return min(lamb, 1.0)

class My_Transform(object):
    '''
    customised image augmentations
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, pth = sample['image'], sample['label'], sample['pth']
        aug = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.RandomRotation(degrees=(0, 180)),
            # transforms.RandomPosterize(bits=2),
            # transforms.RandomAdjustSharpness(sharpness_factor=2),
            # transforms.RandomAutocontrast()
        ])
        augmented = aug(image)

        return {'image': augmented, 'label':label, 'pth':pth}

class My_Normalize(object):
    '''
    Normalise image data
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, pth = sample['image'], sample['label'], sample['pth']
        aug_list = [transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
        aug = transforms.Compose(aug_list)
        # print(image) torch.FloatTensor([label])
        augmented = aug(image)
        return {'image': augmented, 'label':label, 'pth':pth}


def metric(output, labels, batch_size, use_graph=False, use_pool=False):
    p_label = {}
    TP = TN = FP = FN = 0
    slide_label = []
    slide_prob = []

    if use_graph and not use_pool:
        use_patch_prediction = True
    else:
        use_patch_prediction = False

    if use_patch_prediction:
        output = output.view(batch_size, -1, 2)
        for i in range(output.size()[0]):
            prediction_matrix = torch.argmax(output[i], dim=1)
            # prediction = torch.mean(prediction_matrix.float()).int()
            prediction = 1 if 1 in prediction_matrix else 0
            label_use = torch.argmax(labels[i]).cpu().item()
            if prediction == label_use == 0:
                TN += 1
            elif prediction == label_use == 1:
                TP += 1
            elif prediction == 0 and label_use == 1:
                FN += 1
            else:
                FP += 1
    else:
        for i in range(output.size()[0]):
            prediction = torch.argmax(output[i])
            label_use = torch.argmax(labels[i]).cpu().item()
            if prediction.cpu().item() == label_use == 0:
                TN += 1
            elif prediction.cpu().item() == label_use == 1:
                TP += 1
            elif prediction.cpu().item() == 0 and label_use == 1:
                FN += 1
            else:
                FP += 1
            slide_label.append(label_use)
            slide_prob.append(output[i][1].cpu().item())
    return TN, TP, FN, FP




