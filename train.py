import os
import math
import argparse
import logging
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
import cv2


def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def save_metrics(logger, tb_logger, metrics_dict, metrics_name, current_step, opt):
    """
    保存指标到logger和tensorboard
    """
    metrics_avg = sum(metrics_dict.values()) / len(metrics_dict)
    log_s = '# Validation # Mean ' + metrics_name + ': {:.4e}:'.format(metrics_avg)
    # for k, v in metrics_dict.items():
    #     log_s += ' {}: {:.4e}'.format(k, v)
    logger.info(log_s)
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        tb_logger.add_scalar('A-avg_' + metrics_name, metrics_avg, current_step)
        for k, v in metrics_dict.items():
            tb_logger.add_scalar(k, v, current_step)
    pass


def main():
    parser = argparse.ArgumentParser()
    # 正常训练： LOLv1  LOLv2_real LOLv2_synthetic
    parser.add_argument('-opt', type=str, default='./options/train/LOLv1.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # ### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not workf
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        # logging.disable(logging.INFO)
        # logging.disable(logging.WARNING)
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info('You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                print('请安装tensorboardX')
                exit()
                # from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='./tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed'] if opt['train']['manual_seed'] else 2023
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
        del resume_state
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}, total epoch: {:d}'.format(start_epoch, current_step,
                                                                                        total_epochs))
    for epoch in range(start_epoch, total_epochs + 1):
        start_time = time.time()
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                # pbar = util.ProgressBar(len(val_loader))
                psnr_dict = {}
                ssim_dict = {}
                for val_data in val_loader:
                    folder = val_data['folder'][0]
                    idx_d = val_data['idx']
                    # border = val_data['border'].item()

                    # 验证推理
                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()
                    fake_img = util.tensor2img(visuals['fake_img'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

                    # 计算PSNR/SSIM
                    psnr_dict['PSNR-' + folder] = util.calculate_psnr(fake_img, gt_img)
                    if opt['train']['is_val_ssim']:
                        ssim_dict['SSIM-' + folder] = util.calculate_ssim(fake_img, gt_img)

                    # 保存图像
                    if opt['train']['is_save_val_img']:
                        low_img = util.tensor2img(visuals['LQ'])
                        # color_LQ = util.tensor2img(visuals['color_LQ'])  # uint8
                        # fake_color = util.tensor2img(visuals['fake_color'])  # uint8
                        # color_GT = util.tensor2img(visuals['color_GT'])  # uint8
                        # save_img = np.concatenate([low_img, fake_img, gt_img, color_LQ, fake_color, color_GT], axis=1)
                        save_img = np.concatenate([low_img, fake_img, gt_img], axis=1)
                        im_path = os.path.join(opt['path']['val_images'], str(current_step).zfill(6) + '-' +
                                               str(idx_d).split('\'')[1].split('/')[0] + '.png')
                        cv2.imwrite(im_path, save_img.astype(np.uint8))
                    # pbar.update('Test {} - {}'.format(folder, idx_d))
                    pass
                # 保存指标
                save_metrics(logger, tb_logger, psnr_dict, 'PSNR', current_step, opt)
                if opt['train']['is_val_ssim']:
                    save_metrics(logger, tb_logger, ssim_dict, 'SSIM', current_step, opt)
                # exit()
                # 保存模型和训练状态
                avg_psnr = sum(psnr_dict.values()) / len(psnr_dict)
                if rank <= 0:
                    if avg_psnr >= opt['logger']['save_checkpoint_psnr_th']:
                        logger.info('Saving models and training states.')
                        model.save(current_step)
                        model.save_training_state(epoch, current_step)
                    else:
                        logger.info('Not saving models and training states.')
            pass
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
        print('time = ', time.time() - start_time)
    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        tb_logger.close()


if __name__ == '__main__':
    main()
    pass
