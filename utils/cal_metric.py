from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import numpy as np


def psnr_batch(output, gt):
    B = output.shape[0]
    output = output.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
    output[output<0] = 0
    output[output>1] = 1
    gt = gt.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
    gt[gt<0] = 0
    gt[gt>1] = 1
    # output *= 255
    # gt *= 255
    # output = output.astype(np.uint8)
    # gt = gt.astype(np.uint8)
    psnr = 0
    for i in range(B):
        psnr += peak_signal_noise_ratio(gt[i], output[i])
    return psnr/B


def ssim_batch(output, gt):
    B = output.shape[0]
    output = output.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
    output[output < 0] = 0
    output[output > 1] = 1
    gt = gt.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
    gt[gt < 0] = 0
    gt[gt > 1] = 1
    ssim = 0
    # output *= 255
    # gt *= 255
    for i in range(B):
        ssim += structural_similarity(gt[i], output[i], multichannel=True)
    return ssim/B