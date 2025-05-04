import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


##############
class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class VGG19(torch.nn.Module):
    def __init__(self, opt=None, requires_grad=False):
        super().__init__()
        vgg19_model = torchvision.models.vgg19()
        vgg19_model.load_state_dict(torch.load(opt['path']['vgg19_path']))
        vgg_pretrained_features = vgg19_model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, vgg):
        super(VGGLoss, self).__init__()
        self.vgg = vgg
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward2(self, x, y):
        # x: fake, y: real
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward(self, x, y):
        # x: fake, y: real
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss


class L2ContrastiveLoss(nn.Module):
    """
    Compute L2 contrastive loss
    """

    def __init__(self, margin=1, max_violation=False):
        super(L2ContrastiveLoss, self).__init__()
        self.margin = margin

        self.sim = self.l2_sim

        self.max_violation = max_violation

    def forward(self, feature1, feature2):
        # compute image-sentence score matrix
        feature1 = self.l2_norm(feature1)
        feature2 = self.l2_norm(feature2)
        scores = self.sim(feature1, feature2)
        # diagonal = scores.diag().view(feature1.size(0), 1)
        diagonal_dist = scores.diag()
        # d1 = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin - scores).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask.clone()
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]

        loss = (torch.sum(cost_s ** 2) + torch.sum(diagonal_dist ** 2)) / (2 * feature1.size(0))

        return loss

    def l2_norm(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        return x_norm

    def l2_sim(self, feature1, feature2):
        Feature = feature1.expand(feature1.size(0), feature1.size(0), feature1.size(1)).transpose(0, 1)
        return torch.norm(Feature - feature2, p=2, dim=2)


class VGGwithContrastiveLoss(nn.Module):
    def __init__(self, vgg):
        super(VGGwithContrastiveLoss, self).__init__()
        self.vgg = vgg
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='mean')
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        pass

    def forward(self, x, y, z):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        z_vgg_list = []
        for item in z:
            z_vgg = self.vgg(item)
            z_vgg_list.append(z_vgg)
        # 感知损失
        vgg_loss = 0
        for i in range(len(x_vgg)):  # 四个不同尺度的map
            vgg_loss = vgg_loss + self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        # 对比损失
        contra_loss = 0
        for z_vgg in z_vgg_list:
            for i in range(len(x_vgg)):
                contra_loss = contra_loss + self.weights[i] * self.criterion(x_vgg[i], z_vgg[i].detach())
        contra_loss = vgg_loss / (contra_loss / len(z_vgg_list) + 1e-7)
        return contra_loss


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
