
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader



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