import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F



class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        target = target.long()
        index = torch.zeros_like(x, dtype=torch.uint8)
        # print('view size ',target.data.view(-1, 1).size())
        index.scatter_(1, target.data, 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.binary_cross_entropy_with_logits(self.s * output, target.float(), weight=self.weight)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input,
                          target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input,
                 target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        target_tensor = target_tensor.cuda()
        print(target_tensor.size(), target_tensor)
        # print('device check ',input.device, target_tensor.device)
        return self.loss(input, target_tensor)


