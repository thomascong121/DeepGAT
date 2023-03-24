'''
level 5 = (3056, 6912)
no of patch = 500
patch size = 256
comment = non-overlap, discard if < 30% tissure section
steps: 1. sliding window of size 256, crop->224 later
'''


import sys
import os
import json
import cv2
import openslide
import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.patches as patches


class Slide_Patch(object):
    def __init__(self, dimension_level, slide_path, json_path, output_path, minRGB=50, \
                 patch_number=1000, patch_size=256, n_cluster=4, min_tissue=0.3):
        self.dimension_level = dimension_level
        self.slide_path = slide_path
        self.tumor_number = os.path.basename(self.slide_path).split('.')[0]
        self.json_path = json_path
        print('Reading ', self.slide_path)
        self.slide = openslide.OpenSlide(self.slide_path)
        w, h = self.slide.level_dimensions[self.dimension_level]
        self.mask_tumor = np.zeros((h, w))
        self.scale = self.slide.level_downsamples[self.dimension_level]

        self.image_pil = self.slide.read_region((0, 0), self.dimension_level, (w,h)).convert('RGB')
        # self.image_np = np.array(self.image_pil)
        self.img_RGB = np.transpose(np.array(self.slide.read_region((0, 0),
                                                          self.dimension_level,
                                                          (w,h)).convert('RGB')), axes=[1, 0, 2])
        print('pil image size ', self.image_pil.size)
        # self.img_hsv = rgb2hsv(self.img_RGB)
        self.outpath = output_path
        self.minRGB = minRGB
        self.patch_number = patch_number
        self.patch_size = patch_size
        self.n_cluster = n_cluster

        self.out_path = os.path.join(self.outpath, str(self.tumor_number))
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

        self.out_path_normal = os.path.join(self.out_path, 'Normal')
        if not os.path.exists(self.out_path_normal):
            os.mkdir(self.out_path_normal)

        self.out_path_tumor = os.path.join(self.out_path, 'Tumor')
        if not os.path.exists(self.out_path_tumor):
            os.mkdir(self.out_path_tumor)

        self.min_tumor_number = self.patch_size * self.patch_size * min_tissue
        self.thresh_cal()
        self.slide.close()

    def thresh_cal(self):
        print('==> calculate threshold')
        self.color_thresh_R = threshold_otsu(self.img_RGB[:, :, 0])
        self.color_thresh_G = threshold_otsu(self.img_RGB[:, :, 1])
        self.color_thresh_B = threshold_otsu(self.img_RGB[:, :, 2])
        # self.color_thresh_H = threshold_otsu(self.img_hsv[:, :, 1])
        print('==> threshold done')

    def _tissue_mask(self, img=False, check=False, plot=False):
        background_R = self.img_RGB[:, :, 0] > self.color_thresh_R
        background_G = self.img_RGB[:, :, 1] > self.color_thresh_G
        background_B = self.img_RGB[:, :, 2] > self.color_thresh_B
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        # tissue_S = self.img_hsv[:, :, 1] > self.color_thresh_H
        min_R =  self.img_RGB[:, :, 0] > self.minRGB
        min_G =  self.img_RGB[:, :, 1] > self.minRGB
        min_B =  self.img_RGB[:, :, 2] > self.minRGB
        # tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B  ###############tissue mask
        tissue_mask = tissue_RGB & min_R & min_G & min_B###############tissue mask

        if plot:
            plt.figure(0)
            plt.subplot(1, 2, 1)
            plt.imshow(self.img_RGB)
            plt.subplot(1, 2, 2)
            plt.imshow(tissue_mask)
            plt.show()

        return tissue_mask  # levl4

    def _tumor_mask(self, plot=False):
        tumor_json = os.path.basename(self.slide_path).split('.')[0] + '.json'
        tumor_json = os.path.join(self.json_path, tumor_json)

        if not os.path.exists(tumor_json):
            print('not exist')
            tumor_mask = np.array([])
        else:
            with open(tumor_json) as f:
                dicts = json.load(f)
            tumor_polygons = dicts['positive']  # dicts['mask']#

            for tumor_polygon in tumor_polygons:
                # plot a polygon
                name = tumor_polygon["name"]
                vertices = np.array(tumor_polygon["vertices"]) / self.scale
                vertices = vertices.astype(np.int32)
                cv2.fillPoly(self.mask_tumor, [vertices], (255))

            self.mask_tumor = self.mask_tumor[:] > 127
            tumor_mask = np.transpose(self.mask_tumor)
        if plot:
            plt.figure(0)
            plt.subplot(1, 1, 1)
            plt.imshow(self.mask_tumor, cmap='gray')
            plt.show()
        return tumor_mask  # level4

    def mask(self, plot=True):
        tissue_mask = self._tissue_mask(plot=False)
        tumor_mask = self._tumor_mask(plot=False)
        if tumor_mask.shape[0] == 0:
            normal_mask, questionable_mask = tissue_mask, np.zeros((tissue_mask.shape[0], tissue_mask.shape[1]))
        else:
            normal_mask, questionable_mask = tissue_mask & (~ tumor_mask), tissue_mask & (tumor_mask)

        if plot:
            plt.figure(0, figsize=(18, 18))
            plt.subplot(1, 3, 1)
            plt.imshow(normal_mask)
            plt.subplot(1, 3, 2)
            plt.imshow(questionable_mask)
            plt.subplot(1, 3, 3)
            plt.imshow(self.img_RGB)
            plt.show()
        return normal_mask, questionable_mask, self.img_RGB  # level4

    def slide_to_img(self, item_list):
        img, label, coor = item_list

        image = Image.fromarray(img)
        if label == 1:
            img_save_path = os.path.join(self.out_path_normal, 'image')
        else:
            img_save_path = os.path.join(self.out_path_tumor, 'image')
        if not os.path.exists(img_save_path):
            os.mkdir(img_save_path)
        image.save(img_save_path + '/' + str(coor) + '.png')
        # print('==> image saved ',img_save_path)
        return [img_save_path + '/' + str(coor) + '.png', str(label)]

    def obtain_all_patchpts(self):
        '''
        random sampling positive and negative samples
        '''
        normal, questionable_mask, rgb = self.mask(plot=True)  # level4
        # normal
        X_idcs_n, Y_idcs_n = np.where(normal)
        centre_points_normal = np.stack(np.vstack((X_idcs_n.T, Y_idcs_n.T)), axis=1)
        mask_name = [1, 0]
        name = np.full((centre_points_normal.shape[0], 2), mask_name)
        normal_center_points = np.hstack((centre_points_normal, name))
        # tumor
        X_idcs_t, Y_idcs_t = np.where(questionable_mask)
        centre_points_tumor = np.stack(np.vstack((X_idcs_t.T, Y_idcs_t.T)), axis=1)
        mask_name = [0, 1]
        name = np.full((centre_points_tumor.shape[0], 2), mask_name)
        tumor_center_points = np.hstack((centre_points_tumor, name))
        return normal_center_points, tumor_center_points, normal, questionable_mask, rgb  ###########

    def is_tumor(self, x, y, size, tumor_mask):
        if y + size > tumor_mask.shape[0] or x + size > tumor_mask.shape[1]:
            return False
        select_tumor_mask = tumor_mask[y:y + size, x:x + size]
        include_tumor = np.count_nonzero(select_tumor_mask)
        return True if include_tumor / (select_tumor_mask.shape[0] * select_tumor_mask.shape[1]) > 0.01 else False

    def is_normal(self, x, y, size, normal_mask):
        if y + size > normal_mask.shape[0] or x + size > normal_mask.shape[1]:
            return False

        select_normal_mask = normal_mask[y:y + size, x:x + size]
        include_normal = np.count_nonzero(select_normal_mask)
        return True if include_normal / (select_normal_mask.shape[0] * select_normal_mask.shape[1]) > 0.1 else False

    def iter_over_slide(self, x_min, y_min, x_max, y_max, step, level, tm, nm):
        all_sample = []
        rects = []

        img_save_path_normal = os.path.join(self.out_path_normal, 'image')
        img_save_path_tumor = os.path.join(self.out_path_tumor, 'image')

        if not os.path.exists(img_save_path_normal):
            os.mkdir(img_save_path_normal)
        if not os.path.exists(img_save_path_tumor):
            os.mkdir(img_save_path_tumor)

        tumor_count = 0
        normal_count = 0
        # print(x_min, x_max,y_min, y_max, )
        for y in range(y_min, y_max, step):
            for x in range(x_min, x_max, step):

                if self.is_tumor(x, y, step, tm):  # Tumor
                    select_sample = self.img_RGB[y:y + step, x:x + step]
                    image = Image.fromarray(select_sample)
                    image_pth = os.path.join(img_save_path_tumor, str(x) + '_' + str(y) + '.png')
                    image.save(image_pth)
                    mask = tm[y:y + step, x:x + step]
                    mask = Image.fromarray(mask)
                    out_tumor_mask_pth = os.path.join(self.out_path_tumor, 'mask')
                    if not os.path.exists(out_tumor_mask_pth):
                        os.mkdir(out_tumor_mask_pth)
                    mask.save(os.path.join(out_tumor_mask_pth, str(x) + '_' + str(y) + '.png'))
                    rects.append(
                        patches.Rectangle((x, y), self.patch_size, self.patch_size, edgecolor='r', facecolor="none"))
                    tumor_count += 1
                elif self.is_normal(x, y, step, nm):  # Normal
                    select_sample = self.img_RGB[y:y + step, x:x + step]
                    image = Image.fromarray(select_sample)
                    image_pth = os.path.join(img_save_path_normal, str(x) + '_' + str(y) + '.png')
                    image.save(image_pth)
                    rects.append(
                        patches.Rectangle((x, y), self.patch_size, self.patch_size, edgecolor='b', facecolor="none"))
                    normal_count += 1
        print('=> tumor : ', tumor_count)
        print('=> normal : ', normal_count)
        return rects

    def patch_gen(self):
        normal_coord, tumor_coord, normal_mask, tumor_mask, RGB_image = self.obtain_all_patchpts()
        normal_coord = normal_coord[:, 0:2]
        rects = self.iter_over_slide(np.min(normal_coord[:, 1]), np.min(normal_coord[:, 0]), \
                                     np.max(normal_coord[:, 1]), np.max(normal_coord[:, 0]), \
                                     self.patch_size, self.dimension_level, tumor_mask, normal_mask)

        figure, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(RGB_image)
        ax.add_patch(patches.Rectangle((np.min(normal_coord[:, 1]), np.min(normal_coord[:, 0])),
                                       np.max(normal_coord[:, 1]) - np.min(normal_coord[:, 1]),
                                       np.max(normal_coord[:, 0]) - np.min(normal_coord[:, 0]), edgecolor='b',
                                       facecolor="none"))
        for i in rects:
            ax.add_patch(i)
        thumbnail_pth = os.path.join(self.outpath, 'thumbnails')
        if not os.path.exists(thumbnail_pth):
            os.mkdir(thumbnail_pth)
        plot_out = os.path.join(thumbnail_pth, '%s.png'%self.tumor_number)
        plt.savefig(plot_out)
        plt.show()


sldd = '/home/congz3414050/HistoGCN/data/Original/image/tumor_002.tif'
annotation_path = '/home/congz3414050/HistoGCN/data/Original/annotation'
out = '/home/congz3414050/HistoGCN/data/10X'


patch_generator_tumor = Slide_Patch(2, sldd, annotation_path, out, patch_size=256)
patch_generator_tumor.patch_gen()

