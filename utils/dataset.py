from PIL import Image
from torch.utils import data
import os
import torch
import numpy as np
import cv2
from torchvision import transforms


class dataset(data.Dataset):
    def __init__(self, file, nb_of_bin=40, nb_of_frame=12):
        if 'train' not in file and 'test' not in file:
            raise ValueError

        self.file = file
        self.event_file, self.gt_file, self.img_folder = [], [], []
        self.dim = (260, 346)
        self.nb_of_bin = nb_of_bin
        self.nb_of_frame = nb_of_frame
        with open(file, 'r') as f:
            for line in f.readlines():
                p1 = line.strip('\n').split(' ')[0]
                self.event_file.append(p1)
                self.gt_file.append(os.path.join(os.path.split(p1)[0], 'image.jpg'))
                self.img_folder.append(os.path.split(p1)[0])
        self.train = True if 'train' in file else False

    def find_nearest(self, ts, ts_list):
        ts_list = np.array(ts_list)
        dt = ts_list - ts
        dt = np.abs(dt)
        idx = np.where(dt == dt.min())
        return idx[0][0]

    def __getitem__(self, idx):
        gt_img = torch.from_numpy(np.array(Image.open(self.gt_file[idx])).transpose([2,0,1])) / 255
        folder = self.img_folder[idx]
        event_vox = torch.from_numpy(np.load(os.path.join(folder, 'e.npy')))
        img = torch.from_numpy(np.load(os.path.join(folder, 'img.npy'))) / 255
        mask = torch.from_numpy(np.load(os.path.join(folder, 'mask.npy')))
        img, gt_img, event_vox, mask = self.data_augmentation(img, gt_img, event_vox, mask)

        return event_vox, img, gt_img, mask

    def __len__(self):
        return len(self.gt_file)

    def event2vox(self, event):
        event = torch.from_numpy(event).float()
        H, W = self.dim

        voxel_grid = torch.zeros(self.nb_of_bin, H, W, dtype=torch.float32, device='cpu')
        vox = voxel_grid.ravel()

        t, p, x, y = event.t()
        t = t.long()
        time_max = t.max()
        time_min = t.min()

        t = (t-time_min) * (self.nb_of_bin - 1) / (time_max-time_min)
        t = t.float()
        left_t, right_t = t.floor(), t.floor()+1
        left_x, right_x = x.float().floor(), x.float().floor()+1
        left_y, right_y = y.float().floor(), y.float().floor()+1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                for lim_t in [left_t, right_t]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= W-1) & (lim_y <= H-1) & (lim_t <= self.nb_of_bin-1)
                    lin_idx = lim_x.long() + lim_y.long() * W + lim_t.long() * W * H
                    weight = p * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())
                    vox.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

        return voxel_grid

    def event2ecm(self, event):
        event = torch.from_numpy(event).float()
        H, W = self.dim

        voxel_grid = torch.zeros(1, H, W, dtype=torch.float32, device='cpu')
        vox = voxel_grid.ravel()

        t, p, x, y = event.t()
        left_x, right_x = x.float().floor(), x.float().floor()+1
        left_y, right_y = y.float().floor(), y.float().floor()+1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (lim_x <= W-1) & (lim_y <= H-1)
                    lin_idx = lim_x.long() + lim_y.long() * W
                    weight = (1 - (lim_x - x).abs()) * (1 - (lim_y - y).abs())
                    vox.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

        return voxel_grid

    def visEvent(self, events, nb_of_bin, folder, format="%04d.jpg"):
        os.makedirs(folder, exist_ok=True)
        dt = (events[:, 0].max()-events[:, 0].min()) / nb_of_bin
        for k in range(nb_of_bin):
            e1 = events[(events[:, 0] >= k * dt) & (events[:, 0] <= (k + 1) * dt), :]
            ecm = np.zeros([260, 346, 3])
            for e in e1:
                if e[1] == 1:
                    ecm[int(e[3]), int(e[2]), 2] += 1
                else:
                    ecm[int(e[3]), int(e[2]), 0] += 1
            ecm[ecm > 0.8 * ecm.max()] = 0.8 * ecm.max()
            ecm /= ecm.max()
            ecm *= 255
            cv2.imwrite(os.path.join(folder, format % k), ecm)

    def event_grid(self, events):
        events = torch.from_numpy(events)

        t, p, x, y = events.t()
        if min(t.shape) == 0:
            print("Warning")

        t -= t.min()
        time_max = t.max()

        num_voxels = int(2 * np.prod(self.dim) * self.nb_of_bin)
        vox = events[0].new_full([num_voxels, ], fill_value=0)
        H, W = self.dim
        C = self.nb_of_bin

        # normalizing timestamps

        t = t * C/(time_max+1)
        t = t.long()

        # bin = 1 / C
        # s_bin = 0
        # e_bin = bin
        # for i in range(C):
        #     t[(s_bin <= t) == (t < e_bin)] = i
        #     s_bin += bin
        #     e_bin += bin
        # t[-1] = C - 1

        idx = x + W * y + W * H * t + W * H * C * p
        values = torch.full_like(t, 1)

        # draw in voxel grid
        vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(2, C, H, W)
        vox = vox.permute(0,2,3,1)
        # vox = vox.numpy()
        return vox

    def data_augmentation(self, input_image, gt_image, event_grid, mask, crop_size=(256, 256)):
        if isinstance(input_image, np.ndarray):
            input_image = torch.from_numpy(input_image)
        if isinstance(gt_image, np.ndarray):
            gt_image = torch.from_numpy(gt_image)
        if isinstance(event_grid, np.ndarray):
            event_grid = torch.from_numpy(event_grid)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if self.train:
            transforms_list = transforms.Compose([transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip()])
        else:
            transforms_list = transforms.Compose([transforms.CenterCrop(crop_size)])

        img_channel = input_image.shape[0]
        gt_channel = gt_image.shape[0]
        tb = event_grid.shape[-1]
        pos = event_grid[0,...].permute([2,0,1])
        neg = event_grid[1,...].permute([2,0,1])
        x = torch.cat([input_image, gt_image, pos, neg, mask], dim=0)
        x1 = transforms_list(x)
        input_image1 = x1[:img_channel, ...]
        gt_image1 = x1[img_channel:img_channel+gt_channel, ...]
        pos1 = x1[img_channel+gt_channel:img_channel+gt_channel+tb, ...]
        neg1 = x1[img_channel+gt_channel+tb:img_channel+gt_channel+tb+tb, ...]
        mask1 = x1[img_channel+gt_channel+tb+tb:, ...]
        event_grid1 = torch.stack([pos1.permute([1,2,0]), neg1.permute([1,2,0])], dim=0)
        return input_image1, gt_image1, event_grid1, mask1
