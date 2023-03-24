import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor:
    def __init__(self):
        pass

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet50
            """
            # model_ft = models.resnet50(pretrained=use_pretrained)
            model_ft = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 1024
        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def get_extractor(self):
        # Initialize the model for this run
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('using ',device)
        model_ft, input_size = self.initialize_model('resnet', 2, False, use_pretrained=True)
        model_ft.to(device)
        # model_ft.load_state_dict(torch.load(r'F:\BaiduNetdiskDownload\CAMELYON17\normalised_data\Pretrained_ReNEt50_torch_center0.pt'))
        self.set_parameter_requires_grad(model_ft, True) # freeze model
        print(model_ft)
        return model_ft

    def get_classifier(self):
        # Initialize the model for this run
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('using ',device)
        model_ft, input_size = self.initialize_model('resnet', 2, False, use_pretrained=True)
        model_ft.to(device)
        # model_ft.load_state_dict(torch.load(r'F:\BaiduNetdiskDownload\CAMELYON17\normalised_data\Pretrained_ReNEt50_torch_center0.pt'))
        self.set_parameter_requires_grad(model_ft, False) # freeze model
        print(model_ft)
        return model_ft

