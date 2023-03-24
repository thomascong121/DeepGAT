import os
from random import sample


class Sampler:
    def __init__(self, specify_tumor=False):
        self.root = '/home/congz3414050/HistoGCN/data/5X/Tumor'
        if specify_tumor:
            self.tumor_pth = self.root + '/%s'%specify_tumor
        else:
            self.tumor_pth = sample(os.listdir(self.root),1)[0]
        print('Using %s'%self.tumor_pth)
        self.tumor_pth = os.path.join(self.root, self.tumor_pth)
        self.tumor_pth_image = self.tumor_pth + '/Tumor/image'
        self.tumor_pth_mask = self.tumor_pth + '/Tumor/mask'

    def sample(self, num_image):
        tumor_image_folder = os.listdir(self.tumor_pth_image)
        if num_image > len(tumor_image_folder):
            raise Exception('%d num_image is larger than total number of tumor patches')

        tumor_image_pool = sample(tumor_image_folder, num_image)
        out_list = []
        for img_name in tumor_image_pool:
            print('coords ',img_name)
            tumor_image_pth = os.path.join(self.tumor_pth_image, img_name)
            tumor_mask_pth = os.path.join(self.tumor_pth_mask, img_name)
            out_list.append([tumor_image_pth, tumor_mask_pth])
        return out_list

