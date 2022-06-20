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
# path = '/home/lisiqi/code/MIR/log_ablation/2022-05-31/bs12_esai_snn/output/'
# path = '/home/lisiqi/code/MIR/log/2022-04-10/bs16_newCmp+0.050000cmploss_model/output_check/'
# path = '/home/lisiqi/code/MIR/log_ablation/2022-05-31/bs12_esai_snn/output/'
# path = '/home/lisiqi/code/MIR/log_ablation/2022-05-31/bs12_only_event/output/'
# path = '/home/lisiqi/code/MIR/log_ablation/2022-05-31/bs12_only_frame/output/'
# path = '/home/lisiqi/code/MIR/log_ablation/2022-05-31/bs12_wo_event_surface/output/'
# path = '/home/lisiqi/code/MIR/log_ablation/2022-05-31/bs12_wo_fea_fuse/output/'
# path = '/home/lisiqi/code/MIR/log_ablation/2022-05-31/bs12_wo_ms_up/output/'


# path = '/home/lisiqi/code_repo/DeOccNet/log/2022-03-20/bs8/output_DeOccNet/'
# path = '/home/lisiqi/code_repo/E-SAI/codes/log/2022-03-19/bs8/output/'
# path = '/home/lisiqi/code/MIR/log/2022-04-13/bs12_wo_lcmp/output_wo_lcmp/'
# path = '/home/lisiqi/code/MIR/log/2022-04-13/bs12_wo_snn/output_wo_snn/'
path = '/home/lisiqi/code/MIR/log/2022-04-13/bs12_wo_snn_wo_lcmp/output_wo_snn_wo_lcmp/'

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
    op = cv2.imread(os.path.join(path, '%d_output.jpg' % k))
    gt = cv2.imread(os.path.join(path, '%d_gt.jpg' % k))
    # gt = gt.astype(np.float32) / 255.0
    # op = op.astype(np.float32) / 255.0
    p = calpsnr(gt, op)
    s = calssim(gt, op)
    print(k, p, s)
    # break
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
