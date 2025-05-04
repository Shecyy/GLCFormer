import logging
import random
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
from .base_model import BaseModel
from .loss import CharbonnierLoss, CharbonnierLoss2, VGGLoss, VGGwithContrastiveLoss, VGG19

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)
        train_opt = opt['train']

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
            # self.netG = self.netG

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            if train_opt['contrast_weight'] > 0:
                self.adj_table = {}
                self.adj_table_maxsize = train_opt['adj_table_maxsize']
                self.adj_table_item_maxsize = train_opt['adj_table_item_maxsize']
                self.adj_history_negative_ratio = train_opt['adj_history_negative_ratio']
                self.adj_save_ratio = train_opt['adj_save_ratio']
                self.adj_clear_key_ratio = train_opt['adj_clear_key_ratio']
                self.adj_negative_count = train_opt['adj_negative_count']

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'cb2':
                self.cri_pix = CharbonnierLoss2().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']
            self.l_vgg_w = train_opt['vgg_weight']
            self.l_contrast_w = train_opt['contrast_weight']

            if self.l_vgg_w > 0 or self.l_contrast_w > 0:
                vgg = VGG19(opt=opt)
            if self.l_vgg_w > 0:
                self.cri_vgg = VGGLoss(vgg).to(self.device)
            if self.l_contrast_w > 0:
                self.cri_contrast = VGGwithContrastiveLoss(vgg).to(self.device)

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_opt['lr_steps'],
                                                             gamma=train_opt['lr_gamma'])
                    )
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=train_opt['eta_min'])
                    )
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):

        self.var_L = data['LQs'].to(self.device)
        self.light_LQ = data['prior']['light_LQ'].to(self.device)
        self.light_diff_LQ = data['prior']['light_diff_LQ'].to(self.device)
        self.index_list = data['idx']
        self.index_list = [item.split('/')[0] for item in self.index_list]

        if need_GT:
            self.real_H = data['GT'].to(self.device)
        pass

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()
        self.optimizer_G.zero_grad()

        # 前向预测
        self.fake_H = self.netG(self.var_L, self.light_LQ, self.light_diff_LQ)

        # 损失优化
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_final = l_pix

        if self.l_vgg_w > 0:
            l_vgg = self.l_vgg_w * self.cri_vgg(self.fake_H, self.real_H)
            l_final = l_final + l_vgg
        if self.l_contrast_w > 0:
            negative_samples = self.get_negative_samples()
            l_contrast = self.l_contrast_w * self.cri_contrast(self.fake_H, self.real_H, negative_samples)
            l_final = l_final + l_contrast
        l_final.backward()
        self.optimizer_G.step()

        # 损失日志
        self.log_dict['A-l_pix'] = l_pix.item()
        if self.l_vgg_w > 0:
            self.log_dict['A-l_vgg'] = l_vgg.item()
        if self.l_contrast_w > 0:
            self.log_dict['A-l_contrast'] = l_contrast.item()

        if self.l_contrast_w > 0:
            if step % self.opt['train']['clear_step'] == 0:
                self.clear_adj_table()
            self.save_batch_to_table(step)
        pass

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.var_L.shape[2] % 8 != 0:
                clip = self.var_L.shape[2] % 8
                self.var_L = self.var_L[:, :, :-clip, :]
                self.light_LQ = self.light_LQ[:, :, :-clip, :]
                self.real_H = self.real_H[:, :, :-clip, :]
            if self.var_L.shape[3] % 8 != 0:
                clip = self.var_L.shape[3] % 8
                self.var_L = self.var_L[:, :, :, :-clip]
                self.light_LQ = self.light_LQ[:, :, :, :-clip]
                self.real_H = self.real_H[:, :, :, :-clip]

            self.fake_H = self.netG(self.var_L, self.light_LQ, self.light_diff_LQ)
        self.netG.train()

    def get_negative_samples(self):
        negative_samples = [[] for i in range(self.adj_negative_count)]
        count = 0
        for j in range(self.adj_negative_count):
            for i in range(len(self.index_list)):
                key = self.index_list[i]
                if random.random() < self.adj_history_negative_ratio and key in self.adj_table.keys() and len(
                        self.adj_table[key]) > 0:
                    count = count + 1
                    history_enhanced_image = self.adj_table[key].pop()
                    negative_samples[j].append(history_enhanced_image.to(self.device))
                    if len(self.adj_table[key]) == 0:
                        self.adj_table.pop(key)
                    pass
                else:
                    negative_samples[j].append(self.var_L[i].detach().unsqueeze(0))
                    pass
                pass
            pass
        pass
        for i in range(len(negative_samples)):
            negative_samples[i] = torch.cat(negative_samples[i])
        return negative_samples
        pass

    def clear_adj_table(self):
        all_keys = list(self.adj_table.keys())
        for key in all_keys:
            if random.random() < self.adj_clear_key_ratio:
                self.adj_table.pop(key)
        pass

    def save_batch_to_table(self, step):
        for i in range(len(self.index_list)):
            key = self.index_list[i]
            if random.random() < self.adj_save_ratio:  # 保存
                if key in self.adj_table.keys() and len(self.adj_table[key]) < self.adj_table_item_maxsize:
                    image = self.fake_H[i].detach().cpu().unsqueeze(0)
                    self.adj_table[key].add(image)
                elif key not in self.adj_table.keys() and len(self.adj_table.keys()) < self.adj_table_maxsize:
                    image = self.fake_H[i].detach().cpu().unsqueeze(0)
                    self.adj_table[key] = {image}
            pass
        pass

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()

        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['fake_img'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del self.var_L
        del self.fake_H
        del self.real_H

        # del self.light_LQ
        torch.cuda.empty_cache()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
