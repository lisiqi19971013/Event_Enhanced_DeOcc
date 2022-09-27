import numpy as np
import os
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def calpsnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred)


def calssim(gt, pred):
    return structural_similarity(gt, pred, multichannel=True, gaussian_weights=True)


psnrList = []
ssimList = []

path = './log/output'

psnr = 0
ssim = 0
count = 0

psnr_indoor = 0
ssim_indoor = 0
count_indoor = 0

psnr_outdoor = 0
ssim_outdoor = 0
count_outdoor = 0

for k in range(42):
    op = cv2.imread(os.path.join(path, '%d_output.png' % k))
    gt = cv2.imread(os.path.join(path, '%d_gt.png' % k))

    p = calpsnr(gt, op)
    s = calssim(gt, op)
    print(k, p, s)

    psnr += p
    ssim += s
    count += 1

    if k <= 16:
        psnr_indoor += p
        ssim_indoor += s
        count_indoor += 1
    else:
        psnr_outdoor += p
        ssim_outdoor += s
        count_outdoor += 1

psnr /= count
ssim /= count
psnr_indoor /= count_indoor
ssim_indoor /= count_indoor
psnr_outdoor /= count_outdoor
ssim_outdoor /= count_outdoor
print("%f\t%f\t%f\t%f\t%f\t%f" %(psnr, ssim, psnr_indoor, ssim_indoor, psnr_outdoor, ssim_outdoor))

with open(os.path.join(path, 'res1.txt'), 'w') as f:
    f.writelines('PSNR:%f, SSIM:%f\n' % (psnr, ssim))
    f.writelines('PSNR_indoor:%f, SSIM_indoor:%f\n' % (psnr_indoor, ssim_indoor))
    f.writelines('PSNR_outdoor:%f, SSIM_outdoor:%f\n' % (psnr_outdoor, ssim_outdoor))
