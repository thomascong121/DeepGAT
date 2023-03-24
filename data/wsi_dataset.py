import torch
import random
from tqdm.auto import tqdm
import torchvision
import pandas as pd
from util.util import My_Transform, My_Normalize, metric
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class WSIDataset(Dataset):
    """Generate dataset."""
    def __init__(self, filepath, crf=False, annotation_pth=False, down_sample=True, up_sample=False):
        self.all_data = pd.read_csv(filepath)
        self.crf = crf
        self.down_sample = down_sample
        self.up_sample = up_sample
        if not self.crf:
            all_imgpth = self.all_data['image'].tolist()
            all_label = self.all_data['label'].tolist()
            self.X_train, self.X_test, self.y_train, self.y_test = self.stage_split(all_imgpth, all_label, down_sample=down_sample, up_sample=up_sample)

            self.aug_transform = transforms.Compose([My_Transform()])
            self.normalise_transform = transforms.Compose([My_Normalize()])
        else:
            self.annotation = annotation_pth
            all_normal, all_tumor = self.Obtain_detail(self.all_data['label'].tolist())
            print('total: ', all_normal, all_tumor)
            self.train_data = self.all_data.sample(frac=0.8, replace=False, random_state=200)  # random state is a seed value
            self.test_data = self.all_data.drop(self.train_data.index)
            train_normal, train_tumor = self.Obtain_detail(self.train_data['label'].tolist())
            print('train: ', train_normal, train_tumor)
            test_normal, test_tumor = self.Obtain_detail(self.test_data['label'].tolist())
            print('test: ', test_normal, test_tumor)

    def get_cls_num_list(self):
        train_normal, train_tumor = self.Obtain_detail(self.y_train)
        return [train_normal, train_normal]

    def oversample_mixup(self, X, y):
        sythsis_pth = '/home/congz3414050/HistoGCN/data/5X/Tumor/sythesis'
        tumor_list = []
        train_normal, train_tumor = self.Obtain_detail(X, y)
        for i in range(len(y)):
            if y[i] == 'Normal':
                tumor_list.append(X[i])
        num_simluate = len(train_normal) - len(train_tumor)
        lam = np.clip(np.random.beta(0.5, 0.5), 0.4, 0.6)
        print(f'lambda: {lam}')

        # Weighted Mixup
        rst_img = []
        rst_label = []
        print('Rebalancing....')
        for i in tqdm(range(num_simluate)):
            img1_pth = random.sample(tumor_list, 1)[0]
            img2_pth = random.sample(tumor_list, 1)[0]
            source_img = cv2.imread(img1_pth)
            target_img = cv2.imread(img2_pth)
            mixedup_images = lam * source_img + (1 - lam) * target_img
            out_pth = sythsis_pth + '/%d.png' % i
            cv2.imwrite(out_pth, mixedup_images)
            rst_img.append(out_pth)
            rst_label.append('Tumor')
        return rst_img, rst_label

    def stage_split(self, X, y, down_sample, up_sample):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        normal_train, tumor_train = self.Obtain_detail(X_train, y_train)
        n_miniority = len(tumor_train) if len(tumor_train) < len(normal_train) else len(normal_train)
        n_diff = len(normal_train) - len(tumor_train)

        if isinstance(up_sample, str):
            print('Using Upsample....')
            syth_img_folder = os.listdir(up_sample)
            if len(syth_img_folder) != 0:
                # using cached sythesis images
                syth_img = []
                syth_label = []
                for i in range(len(syth_img_folder)):
                    syth_img.append(up_sample + '/' + syth_img_folder[i])
                    syth_label.append('Tumor')
            else:
                # generating sythesis images
                syth_img, syth_label = self.oversample_mixup(self.X_train, self.y_train)

            X_train = X_train + syth_img
            y_train = y_train + syth_label

        elif down_sample:
            print('Using Downsample....')
            normal_train_y = ['Normal' for _ in range(len(normal_train))]
            tumor_train_y = ['Tumor' for _ in range(len(tumor_train))]

            pairs = list(zip(normal_train, normal_train_y))
            pairs = random.sample(pairs, n_miniority)
            normal_train, normal_train_y = zip(*pairs)  # separate the pairs

            X_train = list(normal_train) + tumor_train
            y_train = list(normal_train_y) + tumor_train_y

        print('Train detail: ')
        train_normal, train_tumor = self.Obtain_detail(X_train, y_train)
        print('normal vs tumor = %d vs %d'%(len(train_normal), len(train_tumor)))

        print('Test detail: ')
        test_normal, test_tumor = self.Obtain_detail(X_test, y_test)
        print('normal vs tumor = %d vs %d'%(len(test_normal), len(test_tumor)))

        return X_train, X_test, y_train, y_test

    def Obtain_detail(self, x, y):
        normal =  []
        tumor = []
        for i in range(len(y)):
            if y[i] == 'Normal':
                normal.append(x[i])
            else:
                tumor.append(x[i])
        return normal, tumor

    def Obtain_dataset(self, stage):
        if stage == 'Train':
            if self.crf:
                self.dataset = GridImageDataset(self.train_data, self.annotation, 768, 256)
            else:
                self.dataset = StageDataset(self.X_train, self.y_train, self.normalise_transform, augmentation=self.aug_transform)
        elif stage == 'Test':
            if self.crf:
                self.dataset = GridImageDataset(self.test_data, self.annotation, 768, 256)
            else:
                self.dataset = StageDataset(self.X_test, self.y_test, self.normalise_transform)
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
    def __init__(self, X, y, normalisation, augmentation=False):
        self.map_dict = {'Normal': torch.FloatTensor([1.0, 0.0]), 'Tumor': torch.FloatTensor([0.0, 1.0])}
        self.X = X
        self.y = y
        self.normalisation = normalisation
        self.augmentation = augmentation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        '''image	label'''
        img_path = self.X[idx]
        label = self.map_dict[self.y[idx]]
        # image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        image = Image.open(img_path)
        sample = {'image': image, 'label': label, 'pth':img_path}

        if self.augmentation:
            sample = self.augmentation(sample)
        sample = self.normalisation(sample)
        return sample


class GridImageDataset(Dataset):
    """
    Data producer that generate a square grid, e.g. 3x3, of patches and their
    corresponding labels from pre-sampled images.
    """

    def __init__(self, data_path, json_path, img_size, patch_size,
                 crop_size=224, normalize=True):
        """
        Initialize the data producer.
        Arguments:
            data_path: string, path to pre-sampled images using patch_gen.py
            img_size: int, size of pre-sampled images, e.g. 768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
        """
        self._df = data_path
        self._json_path = json_path
        self._img_size = img_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)
        self._preprocess()

    def _preprocess(self):
        if self._img_size % self._patch_size != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(self._img_size, self._patch_size))

        self._patch_per_side = self._img_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

        self._pids = list(map(lambda x: x.strip('.json'),
                              os.listdir(self._json_path)))

    #         self._annotations = {}
    #         for pid in self._pids:
    #             pid_json_path = os.path.join(self._json_path, pid + '.json')
    #             anno = Annotation()
    #             anno.from_json(pid_json_path)
    #             self._annotations[pid] = anno

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        image_pth = self._df.iloc[idx, 0]
        mask_pth = self._df.iloc[idx, 3]

        slide_name = self._df.iloc[idx, 2]
        image_name = image_pth.split('/')[-1].split('.png')[0].split('_')
        x_top_left, y_top_left = 0, 0  # int(image_name[0]), int(image_name[1])
        patch_label = self._df.iloc[idx, 1]

        # the grid of labels for each patch
        label_grid = np.zeros((self._patch_per_side, self._patch_per_side),
                              dtype=np.float32)
        if mask_pth != 'Nothing':
            mask_np = cv2.imread(mask_pth)
            for x_i in range(self._patch_per_side):
                x_t = x_top_left + self._patch_size * x_i
                for y_i in range(self._patch_per_side):
                    y_t = y_top_left + self._patch_size * y_i
                    select_tumor_mask = mask_np[x_t: x_t + self._patch_size, y_t: y_t + self._patch_size]
                    include_tumor = np.count_nonzero(select_tumor_mask)

                    if include_tumor / (mask_np.shape[0] * mask_np.shape[1]) > 0.001:
                        label = 1
                    else:
                        label = 0

                    label_grid[y_i, x_i] = label

        img = Image.open(image_pth)

        # color jitter
        img = self._color_jitter(img)

        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label_grid = np.fliplr(label_grid)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)
        label_grid = np.rot90(label_grid, num_rotate)

        # PIL image:   H x W x C
        # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        # flatten the square grid
        img_flat = np.zeros(
            (self._grid_size, 3, self._crop_size, self._crop_size),
            dtype=np.float32)
        label_flat = np.zeros(self._grid_size, dtype=np.float32)

        idx = 0
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # center crop each patch
                x_start = int(
                    (x_idx + 0.5) * self._patch_size - self._crop_size / 2)
                x_end = x_start + self._crop_size
                y_start = int(
                    (y_idx + 0.5) * self._patch_size - self._crop_size / 2)
                y_end = y_start + self._crop_size
                img_flat[idx] = img[:, x_start:x_end, y_start:y_end]
                label_flat[idx] = label_grid[x_idx, y_idx]

                idx += 1

        return (img_flat, label_flat)