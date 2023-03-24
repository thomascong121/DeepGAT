
from My_cams.


import GradCAM,ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from My_cams.pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision
from util import metric, get_adj
from torch_geometric.data import Data, DataLoader
from network.ResGNN import ResGNN
from network.DenseGNN import DenseGNN
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import torch.nn as nn
from tumor_pool import Sampler
from PIL import Image
import matplotlib.pyplot as plt


class Cam_Visuals:
    def __init__(self, model_name, in_feature, h_feature, our_model, cnn_model):
        our_load_pth = '/home/congz3414050/HistoGCN/checkpoint/Scratch_%sGAT_torch.pt'%model_name
        cnn_load_pth = '/home/congz3414050/HistoGCN/checkpoint/Scratch_%s_torch.pt'%model_name
        aug_list = [transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
        self.aug = transforms.Compose(aug_list)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # define CNN+GAT
        self.ourmodel = our_model
        self.ourmodel.load_state_dict(torch.load(our_load_pth))
        self.target_layers_ours = [self.ourmodel.feature_et[-1][-1]]#[self.ourmodel.gcn1]#
        print(self.target_layers_ours)
        print('==========================')
        self.ourmodel.eval()

        # define CNN
        self.cnnmodel = cnn_model
        if model_name == 'Dense':
            self.cnnmodel.classifier = nn.Linear(h_feature, 2)
            self.target_layers_cnn = [self.cnnmodel.features[-2]]
        elif model_name == 'ReNet50' or model_name == 'ReNet34':
            num_ftrs = self.cnnmodel.fc.in_features
            self.cnnmodel.fc = nn.Linear(num_ftrs, 2)
            self.target_layers_cnn = [self.cnnmodel.layer4[-1]]
        print(self.target_layers_cnn)
        self.cnnmodel.load_state_dict(torch.load(cnn_load_pth))
        self.cnnmodel.eval()

        self.Sampler = Sampler()#specify_tumor='tumor_101'
    def get_visual(self):
        sample_list = self.Sampler.sample(1)
        adj_matrx = get_adj(8)
        adj_matrx = np.array(adj_matrx)
        adj_matrx = torch.tensor(adj_matrx, dtype=torch.long)
        adj_matrx = adj_matrx.t().contiguous()
        data_list = [Data(x=torch.rand(1, 1), edge_index=adj_matrx)]
        graphloader = DataLoader(data_list, batch_size=1)

        for i in range(len(sample_list)):
            img_pth, mask_pth = sample_list[i][0], sample_list[i][1]
            rgb_img = cv2.imread(img_pth, 1)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255
            img_pil = Image.open(img_pth)
            mask_pil = Image.open(mask_pth)

            input_tensor = self.aug(img_pil)
            input_tensor = input_tensor.unsqueeze(0)
            cam_our = GradCAM(model=self.ourmodel, target_layers=self.target_layers_ours, use_cuda=True)
            cam_cnn = GradCAM(model=self.cnnmodel, target_layers=self.target_layers_cnn, use_cuda=True)
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            for graph_data in graphloader:
                grayscale_cam_our = cam_our(input_tensor=(input_tensor, graph_data))
            grayscale_cam_cnn = cam_cnn(input_tensor=input_tensor)

            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam_our = grayscale_cam_our[0, :]
            grayscale_cam_cnn = grayscale_cam_cnn[0, :]
            visualization_our = show_cam_on_image(rgb_img, grayscale_cam_our)
            visualization_cnn = show_cam_on_image(rgb_img, grayscale_cam_cnn)

            plt.figure(figsize=(20,10))
            plt.subplot(1,4,1)
            plt.imshow(img_pil)
            plt.subplot(1,4,2)
            plt.imshow(mask_pil)
            plt.subplot(1, 4, 3)
            plt.imshow(visualization_our)
            plt.subplot(1, 4, 4)
            plt.imshow(visualization_cnn)
            plt.show()


if __name__ == '__main__':
    input_ft = 2048
    h_ft = 1024
    our_model = ResGNN(input_ft, h_ft, 2, 0.5)
    cnn_model = torchvision.models.resnet50()
    visuals = Cam_Visuals('ReNet50', input_ft, h_ft, our_model, cnn_model)
    visuals.get_visual()

    '''
    tumor_101 1537_12033.png
    tumor_074 9472_3072.png
    '''