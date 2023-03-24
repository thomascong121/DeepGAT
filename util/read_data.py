import torch
from tqdm.auto import tqdm
from torch_geometric.data import Data, Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
import numpy as np
import pandas as pd
# from pydist2.distance import pdist2
from util.load_pretrained import FeatureExtractor
from util.util import My_Normalize

print(torch.__version__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
activation = {}
def get_act(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class WSIDataset(Dataset):
    def __init__(self, root, checkpoint_dir, transform=None, pre_transform=None):
        self.root_fold = os.path.join(root, 'csvs')
        self.root_fold_dir = os.listdir(self.root_fold)
        self.root_fold_dir.sort()
        self.label_map = {'Normal': 0.0, 'Tumor': 1.0}
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.ft_ext = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        num_ftrs = self.ft_ext.fc.in_features
        # print(self.ft_ext.layer4.size())
        # print(self.ft_ext)
        self.ft_ext.fc = nn.Linear(num_ftrs, 2)
        self.ft_ext.load_state_dict(
            torch.load(checkpoint_dir + '/Pretrained_ReNet50_torch.pt'))
        self.ft_ext.to(device)
        self.ft_ext.avgpool.register_forward_hook(get_act('avgpool'))
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        raw_file = []
        for tum_csv in self.root_fold_dir:
            tum_csv_pth = os.path.join(self.root_fold, tum_csv)
            raw_file.append(tum_csv_pth)
        return raw_file

    @property
    def processed_file_names(self):
        processed = []
        if not os.path.exists((self.processed_dir)):
            return 'nothing'

        for f in os.listdir(self.processed_dir):
            processed.append(f)
        return processed
        # return 'nothing'

    def download(self):
        pass

    def process(self):
        all_tumor_csvs = self.raw_paths
        for csv in tqdm(all_tumor_csvs):
            csv_name = csv.split('\\')[-1].split('.csv')[0]
            tumor_number = csv_name.split('Tumor_')[-1]
            current = pd.read_csv(csv)
            current_node_ft = self._obtain_node_feature(current)
            current_adj_matr, current_edge_wt = self._obtain_adj_matrx(current)
            current_node_label = self._obtain_node_label(current)

            # create dataset
            data = Data(x=current_node_ft,
                        edge_index=current_adj_matr,
                        edge_attr=current_edge_wt,
                        y=current_node_label)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, '%s.pt' % csv_name))
            # break

    def _obtain_feature(self, img_pth):
        aug_list = [transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
        aug = transforms.Compose(aug_list)

        img_pil = Image.open(img_pth).convert('RGB')
        img_normalised = aug(img_pil)
        output = self.ft_ext(img_normalised.unsqueeze(0).cuda())
        ff = torch.flatten(activation['avgpool'])
        return ff.cpu()

    def _obtain_node_feature(self, df):
        node_features = []

        for i in range(len(df)):
            node_pth = df.iloc[i, 0]
            node_feature = self._obtain_feature(node_pth)
            node_features.append(node_feature)

        node_features = torch.stack(node_features)
        return node_features

    def _obtain_adj_matrx(self, df):
        adj_matrx = []
        edge_wt = []
        for i in range(len(df)):
            i_coord = df.iloc[i, 1]
            i_id = df.iloc[i, 3]
            i_x, i_y = int(i_coord.split('_')[0]), int(i_coord.split('_')[1])
            i_x, i_y = i_x + 128, i_y + 128
            for j in range(len(df)):
                if i != j:
                    j_coord = df.iloc[j, 1]
                    j_id = df.iloc[j, 3]
                    j_x, j_y = int(j_coord.split('_')[0]), int(j_coord.split('_')[1])
                    j_x, j_y = j_x + 128, j_y + 128
                    if abs(j_x - i_x) <= 256 and abs(j_y - i_y) <= 256:
                        node_i_feature = self._obtain_feature(df.iloc[i, 0])
                        node_j_feature = self._obtain_feature(df.iloc[j, 0])
                        cos_sim = self.cos_sim(node_i_feature.unsqueeze(0), node_j_feature.unsqueeze(0))
                        connect = [i_id, j_id]
                        adj_matrx.append(connect)
                        edge_wt.append([cos_sim])
        adj_matrx = np.array(adj_matrx)
        adj_matrx = torch.tensor(adj_matrx, dtype=torch.long)
        edge_wt = np.array(edge_wt)
        edge_wt = torch.tensor(edge_wt)
        return adj_matrx.t().contiguous(), edge_wt

    def _obtain_node_label(self, df):
        node_label = []
        for i in range(len(df)):
            label = df.iloc[i, 2]
            node_label.append(self.label_map[label])
        node_label = np.array(node_label)
        return torch.tensor(node_label, dtype=torch.float)

    def len(self):
        return len(self.processed_file_names) - 2

    def get(self, idx):
        tumor_idx = self.root_fold_dir[idx]
        tumor_name = tumor_idx.split('.csv')[0]
        print('check ',tumor_name)
        data = torch.load(os.path.join(self.processed_dir, '%s.pt' % tumor_name))#torch.load(os.path.join(self.processed_dir, 'Tumor_082.pt'))#
        return data