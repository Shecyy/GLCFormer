import os.path as osp
import logging
import argparse
import torch.cuda
import options.options as option
import utils.util as util
import cv2
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
# LOLv1  LOLv2_real LOLv2_synthetic
parser.add_argument('-opt', type=str, default='./options/test/LOLv1.yml')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main():
    save_imgs = True
    model = create_model(opt)
    save_folder = './results/{}'.format(opt['name'])
    GT_folder = osp.join(save_folder, 'images/GT')
    output_folder = osp.join(save_folder, 'images/output')
    input_folder = osp.join(save_folder, 'images/input')
    util.mkdirs(save_folder)
    util.mkdirs(GT_folder)
    util.mkdirs(output_folder)
    util.mkdirs(input_folder)

    print('mkdir finish')

    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))
        psnr_rlt = {}  # with border and center frames
        psnr_rlt_avg = {}
        psnr_total_avg = 0.

        ssim_rlt = {}  # with border and center frames
        ssim_rlt_avg = {}
        ssim_total_avg = 0.

        for val_index, val_data in enumerate(val_loader):
            for kkk in range(4):
                torch.cuda.empty_cache()

            folder = val_data['folder'][0]
            print(folder)
            idx_d = val_data['idx']
            if psnr_rlt.get(folder, None) is None:
                psnr_rlt[folder] = []

            if ssim_rlt.get(folder, None) is None:
                ssim_rlt[folder] = []
            try:
                model.feed_data(val_data)
                model.test()
            except:
                print('Exception = ', folder)
                continue
                pass
            visuals = model.get_current_visuals()
            fake_img = util.tensor2img(visuals['fake_img'])  # uint8
            gt_img = util.tensor2img(visuals['GT'])  # uint8

            mid_ix = dataset_opt['N_frames'] // 2
            input_img = util.tensor2img(visuals['LQ'])

            if save_imgs:
                try:
                    tag = '{}.{}'.format(val_data['folder'], idx_d[0].replace('/', '-'))
                    print(osp.join(output_folder, '{}.png'.format(tag)))
                    cv2.imwrite(osp.join(output_folder, '{}.png'.format(tag)), fake_img)
                    cv2.imwrite(osp.join(GT_folder, '{}.png'.format(tag)), gt_img)
                    cv2.imwrite(osp.join(input_folder, '{}.png'.format(tag)), input_img)
                except Exception as e:
                    print(e)
                    import ipdb;
                    ipdb.set_trace()

            # calculate PSNR
            psnr = util.calculate_psnr(fake_img, gt_img)
            # psnr = 0
            psnr_rlt[folder].append(psnr)

            ssim = util.calculate_ssim(fake_img, gt_img)
            # ssim = 0
            ssim_rlt[folder].append(ssim)
            pbar.update('Test {} - {}'.format(folder, idx_d))

        for k, v in psnr_rlt.items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]

        for k, v in ssim_rlt.items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]

        psnr_total_avg /= len(psnr_rlt)
        ssim_total_avg /= len(ssim_rlt)
        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
        for k, v in psnr_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
        for k, v in ssim_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        psnr_all = 0
        psnr_count = 0
        for k, v in psnr_rlt.items():
            psnr_all += sum(v)
            psnr_count += len(v)
        psnr_all = psnr_all * 1.0 / psnr_count
        print(psnr_all)

        ssim_all = 0
        ssim_count = 0
        for k, v in ssim_rlt.items():
            ssim_all += sum(v)
            ssim_count += len(v)
        ssim_all = ssim_all * 1.0 / ssim_count
        print(ssim_all)
        pass
    pass


if __name__ == '__main__':
    main()
    pass
