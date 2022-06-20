import sys
import numpy as np
sys.path.append('..')
from utils.dataset import dataset as dataset
from model.model_full import model as M
import cv2


def saveImg(img, path):
    img[img > 1] = 1
    img[img < 0] = 0
    op = np.zeros([img.shape[2], img.shape[3], 3])
    img = img[0].cpu().permute(1,2,0) * 255
    op[:,:,0] = img[:,:,2]
    op[:,:,1] = img[:,:,1]
    op[:,:,2] = img[:,:,0]
    cv2.imwrite(path, op)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    import torch
    from torch import nn
    from PIL import Image
    from utils import psnr_batch, ssim_batch
    import cv2

    device = 'cuda'

    run_dir = '/home/lisiqi/code/MIR/log/2022-04-10/bs16_newCmp+0.050000cmploss_model/'
    # run_dir = '/home/lisiqi/code/MIR/log/2022-04-12/bs12_wo_lcmp/'
    # run_dir = '/home/lisiqi/code/MIR/log/2022-04-13/bs12_wo_lcmp/'

    testFolder = dataset('/home/lisiqi/data/SAIdata/test.txt')
    testLoader = torch.utils.data.DataLoader(testFolder, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    model = M(netParams={'Ts': 1, 'tSample': 40}, inChannels=33+11, norm='BN')
    model = torch.nn.DataParallel(model)
    model.cuda()

    '''frame'''
    # a = torch.load('/home/lisiqi/code/E-DeOcc/log/bs9_FrameDeOcc_light/'+'ckpt.pth.tar')
    # model.FrameEncoder.MaskPredNet.load_state_dict(a['MaskPredNet'])
    # model.FrameEncoder.Encoder.load_state_dict(a['Encoder'])
    # model.FrameDecoder.load_state_dict(a['Decoder'])

    '''event+frame'''
    # a = torch.load(run_dir+'checkpoint.pth.tar')
    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(a['state_dict'])

    '''full'''
    print('==> loading existing model:', os.path.join(run_dir, 'checkpoint.pth.tar'))
    model_info = torch.load(os.path.join(run_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(model_info['state_dict'])

    savePath = os.path.join(run_dir, 'output_check')
    os.makedirs(savePath, exist_ok=True)

    with torch.no_grad():
        model.eval()
        psnr = 0
        ssim = 0
        count = 0

        psnr_indoor = 0
        ssim_indoor = 0
        count_indoor = 0

        psnr_outdoor = 0
        ssim_outdoor = 0
        count_outdoor = 0
        for i, (event_vox, img, gt_img, mask) in enumerate(testLoader):

            event_vox = event_vox.cuda()
            img = img.cuda().float()
            mask = mask.cuda().float()
            gt_img = gt_img.cuda().float()

            mask = torch.index_select(mask, 1, torch.tensor([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31]).cuda())
            output = model(event_vox, torch.cat([img, mask], dim=1))

            # for k in range(11):
            #     img1 = outputMask[0, k*3, :, :] * 255
            #     img1[img1>255] = 255
            #     img1[img1 <0] = 0
            #     Image.fromarray(np.array(img1.cpu()).astype(np.uint8)).save(os.path.join(path, '%d_mask%d.jpg' % (i, k)))
            # for k in range(11):
            #     img1 = mask[0, k*3, :, :] * 255
            #     img1[img1>255] = 255
            #     img1[img1 <0] = 0
            #     Image.fromarray(np.array(img1.cpu()).astype(np.uint8)).save(os.path.join(path, '%d_mask%d_init.jpg' % (i, k)))
            # for k in range(11):
            #     img1 = img[0, k*3:k*3+3, :, :].permute(1,2,0) * 255
            #     img1[img1>255] = 255
            #     img1[img1 <0] = 0
            #     Image.fromarray(np.array(img1.cpu()).astype(np.uint8)).save(os.path.join(path, '%d_img%d.jpg' % (i, k)))

            # for k in range(36):
            #     e = event_with_refocus[0, k, :, :]
            #     e -= e.min()
            #     e /= e.max()
            #     img1 = e * 255
            #     img1[img1>255] = 255
            #     img1[img1 <0] = 0
            #     Image.fromarray(np.array(img1.cpu()).astype(np.uint8)).save(os.path.join(path, '%d_event%d.jpg' % (i, k)))

            # output = model(event_vox, img, mask)
            p = psnr_batch(output, gt_img)
            s = ssim_batch(output, gt_img)

            saveImg(output, os.path.join(savePath, '%d_output.png'%i))
            saveImg(gt_img, os.path.join(savePath, '%d_gt.png'%i))
            # output[output>1] = 1
            # output[output<0] = 0
            # output = output[0].permute(1,2,0) * 255
            # gt_img = gt_img[0].permute(1,2,0) * 255
            # cv2.imwrite()
            # Image.fromarray(np.array(output.detach().cpu()).astype(np.uint8)).save(os.path.join(savePath, '%d_output.jpg'%i))
            # Image.fromarray(np.array(gt_img.detach().cpu()).astype(np.uint8)).save(os.path.join(savePath, '%d_gt.jpg'%i))

            print(i, p, s)
            psnr += p
            ssim += s
            count += 1

            if i <= 16:
                psnr_indoor += p
                ssim_indoor += s
                count_indoor += 1
            else:
                psnr_outdoor += p
                ssim_outdoor += s
                count_outdoor += 1
            # break

        psnr /= count
        ssim /= count
        print(psnr, ssim, psnr_indoor/count_indoor, ssim_indoor/count_indoor, psnr_outdoor/count_outdoor, ssim_outdoor/count_outdoor)

# with open(os.path.join(run_dir, 'output','res.txt'), 'w') as f:
#     f.writelines('PSNR:%f, SSIM:%f\n'%(psnr, ssim))
#     f.writelines('PSNR_indoor:%f, SSIM_indoor:%f\n'%(psnr_indoor/count_indoor, ssim_indoor/count_indoor))
#     f.writelines('PSNR_outdoor:%f, SSIM_outdoor:%f\n'%(psnr_outdoor/count_outdoor, ssim_outdoor/count_outdoor))