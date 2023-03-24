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


class CB_loss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, beta):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        super(CB_loss, self).__init__()
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        print('E ',effective_num, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        print('inverse E ',weights)
        self.weights = weights / np.sum(weights) * no_of_classes
        self.no_of_classes = no_of_classes
        print('CB weights ',self.weights)
        # labels_one_hot = F.one_hot(labels, no_of_classes).float()

    def forward(self,logits, labels):
        weights = torch.tensor(self.weights).float().cuda()
        return F.binary_cross_entropy_with_logits(input=logits, target=labels, weight=weights)


if __name__ == '__main__':
    no_of_classes = 2
    logits = torch.rand(5, no_of_classes).float().cuda()
    labels = torch.tensor([[0,1],[1,0],[1,0],[0,1],[0,1]]).cuda()
    beta = 0.99
    samples_per_cls = [100, 10]
    print('sample: ',logits.size(), labels.size())
    LDAM_loss = LDAMLoss(samples_per_cls)
    loss = LDAM_loss(logits, labels.float())
    print('loss is ',loss)