import openslide
import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from tqdm.auto import tqdm
from collections import defaultdict
from annotation import Annotation
import os


fold = r'F:\Cong\NCRF-master\coords'
anno_pth = r'F:\BaiduNetdiskDownload\CAMELYON16\training\annotation_json'
out_pth = r'F:\BaiduNetdiskDownload\CAMELYON16\GCN\data\40X\data'
root_path = r'F:\BaiduNetdiskDownload\CAMELYON16\training\tumor'
tumor_dict = defaultdict(list)

for fl in os.listdir(fold):
    fl_pth = os.path.join(fold, fl)
    file1 = open(fl_pth, 'r')
    lines = file1.readlines()
    for line in tqdm(lines):
        line_list = line.split(',')
        img_name = line_list[0]
        x,y = line_list[1],line_list[2]
        img_type, number = img_name.split('_')[0], img_name.split('_')[1]
        if img_type == 'Tumor':
            tumor_dict[str(number)].append([x,y])

for k in sorted(list(tumor_dict.keys())):
    print('tumor %s has %d'%(k, len(tumor_dict[k])))

class patch_gen:
    def __init__(self, tumor_dict, root_path, annotation_pth, out_pth, patch_size):
        self.tumor_dict = tumor_dict
        self.root_path = root_path
        self.annotation_pth = annotation_pth
        self.out_pth = out_pth
        self.patch_size = patch_size

    def loop_over_slide(self):
        tumor_st = defaultdict(list)
        for nb in self.tumor_dict.keys():
            tumor_nb_wotif = 'Tumor_%s'%nb
            tumor_nb_wotif_nocap = 'tumor_%s.tif'%nb
            tumor_path = os.path.join(self.root_path, tumor_nb_wotif_nocap)
            current_slide = openslide.OpenSlide(tumor_path)
            out_pth_tumor_slide = os.path.join(self.out_pth, tumor_nb_wotif)
            if not os.path.exists(out_pth_tumor_slide):
                os.mkdir(out_pth_tumor_slide)
            out_tumor = os.path.join(out_pth_tumor_slide, 'Tumor')
            if not os.path.exists(out_tumor):
                os.mkdir(out_tumor)
            out_normal = os.path.join(out_pth_tumor_slide, 'Normal')
            if not os.path.exists(out_normal):
                os.mkdir(out_normal)
            out_tumor_img = os.path.join(out_tumor, 'Image')
            if not os.path.exists(out_tumor_img):
                os.mkdir(out_tumor_img)
            out_normal_img = os.path.join(out_normal, 'Image')
            if not os.path.exists(out_normal_img):
                os.mkdir(out_normal_img)

            annotation_json = os.path.join(self.annotation_pth, tumor_nb_wotif + '.json')
            anno = Annotation()
            anno.from_json(annotation_json)
            corrds = self.tumor_dict[nb]
            normal, tumor = 0, 0
            for coord in tqdm(corrds):
                x_center, y_center = int(coord[0]), int(coord[1])

                x_top_left = int(x_center - self.patch_size / 2)
                y_top_left = int(y_center - self.patch_size / 2)
                patch = current_slide.read_region((x_top_left, y_top_left), 0, (self.patch_size, self.patch_size))
                if anno.inside_polygons((x_center, y_center), True):
                    tumor += 1
                    patch.save(os.path.join(out_tumor_img, '%d_%d.png'%(x_top_left, y_top_left)))
                else:
                    normal += 1
                    patch.save(os.path.join(out_normal_img, '%d_%d.png' % (x_top_left, y_top_left)))
            tumor_st[nb] = [normal, tumor]
            # break
            current_slide.close()
        return tumor_st

# patch_gen = patch_gen(tumor_dict, root_path, anno_pth, out_pth, 256)
# tum_st = patch_gen.loop_over_slide()
# print('===')
# for k in sorted(list(tum_st.keys())):
#     print('tumor %s has %s'%(k, str(tum_st[k])))



'''
20 ---
tumor 010 has [0, 2000]
tumor 015 has [0, 2000]
tumor 018 has [0, 2000]
tumor 020 has [0, 2000]
tumor 025 has [0, 2000]
tumor 029 has [0, 2000]
tumor 033 has [0, 2000]
tumor 034 has [0, 2000]
tumor 044 has [0, 2000]
tumor 046 has [0, 2000]
tumor 051 has [0, 2000]
tumor 054 has [0, 2000]
tumor 055 has [0, 2000]
tumor 056 has [0, 2000]
tumor 067 has [0, 2000]
tumor 079 has [0, 2000]
tumor 085 has [0, 2000]
tumor 092 has [0, 2000]
tumor 095 has [0, 2000]
tumor 110 has [0, 2000]

Process finished with exit code 0
tumor 001 has [1323, 2000]
tumor 002 has [1212, 2000]
tumor 003 has [1065, 2000]
tumor 004 has [1090, 2000]
tumor 005 has [607, 2000]
tumor 006 has [2285, 2000]
tumor 007 has [1373, 2000]
tumor 008 has [1407, 2000]
tumor 009 has [1968, 2000]
tumor 011 has [1834, 2000]
tumor 012 has [1021, 2000]
tumor 013 has [1337, 2000]
tumor 014 has [1025, 2000]
tumor 016 has [939, 2000]
tumor 017 has [661, 2000]
tumor 019 has [850, 2000]
tumor 021 has [1094, 2000]
tumor 022 has [1705, 2000]
tumor 023 has [732, 2000]
tumor 024 has [1046, 2000]
tumor 026 has [2164, 2000]
tumor 027 has [1259, 2000]
tumor 028 has [1666, 2000]
tumor 030 has [1093, 2000]
tumor 031 has [1817, 2000]
tumor 032 has [1376, 2000]
tumor 035 has [738, 2000]
tumor 036 has [2499, 2000]
tumor 037 has [1585, 2000]
tumor 038 has [910, 2000]
tumor 039 has [794, 2000]
tumor 040 has [2021, 2000]
tumor 041 has [832, 2000]
tumor 042 has [1517, 2000]
tumor 043 has [700, 2000]
tumor 045 has [1373, 2000]
tumor 047 has [1231, 2000]
tumor 048 has [1307, 2000]
tumor 049 has [718, 2000]
tumor 050 has [671, 2000]
tumor 052 has [2210, 2000]
tumor 053 has [1045, 2000]
tumor 057 has [844, 2000]
tumor 058 has [1048, 2000]
tumor 059 has [606, 2000]
tumor 060 has [1198, 2000]
tumor 061 has [3720, 2000]
tumor 062 has [1658, 2000]
tumor 063 has [2036, 2000]
tumor 064 has [1396, 2000]
tumor 065 has [766, 2000]
tumor 066 has [1280, 2000]
tumor 068 has [1964, 2000]
tumor 069 has [2803, 2000]
tumor 070 has [864, 2000]
tumor 071 has [1223, 2000]
tumor 072 has [1165, 2000]
tumor 073 has [957, 2000]
tumor 074 has [1002, 2000]
tumor 075 has [1757, 2000]
tumor 076 has [3684, 2000]
tumor 077 has [3175, 2000]
tumor 078 has [1806, 2000]
tumor 080 has [887, 2000]
tumor 081 has [664, 2000]
tumor 082 has [1114, 2000]
tumor 083 has [1248, 2000]
tumor 084 has [2540, 2000]
tumor 086 has [1343, 2000]
tumor 087 has [892, 2000]
tumor 088 has [1464, 2000]
tumor 089 has [9456, 2000]
tumor 090 has [1185, 2000]
tumor 091 has [1000, 2000]
tumor 093 has [867, 2000]
tumor 094 has [5474, 2000]
tumor 096 has [1646, 2000]
tumor 097 has [1808, 2000]
tumor 098 has [731, 2000]
tumor 099 has [1337, 2000]
tumor 100 has [1167, 2000]
tumor 101 has [1038, 2000]
tumor 102 has [3915, 2000]
tumor 103 has [860, 2000]
tumor 104 has [2580, 2000]
tumor 105 has [914, 2000]
tumor 106 has [1586, 2000]
tumor 107 has [585, 2000]
tumor 108 has [881, 2000]
tumor 109 has [866, 2000]
'''