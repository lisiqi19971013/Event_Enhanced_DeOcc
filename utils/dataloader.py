import torch


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.event_vox, self.img, self.mask, self.gt_img = next(self.loader)

        except StopIteration:
            self.event_vox = None
            self.img = None
            self.mask = None
            self.gt_img = None
            return

        with torch.cuda.stream(self.stream):
            self.event_vox = self.event_vox.cuda(non_blocking=True)
            self.img = self.img.cuda(non_blocking=True)
            self.mask = self.mask.cuda(non_blocking=True)
            self.gt_img = self.gt_img.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        event_vox = self.event_vox
        img = self.img
        mask = self.mask
        gt_img = self.gt_img
        self.preload()
        # return view, pcs, pc_parts
        return event_vox, img, mask, gt_img
    

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    from dataset import dataset
    trainFolder = dataset('/home/lisiqi/data/SAIdata/train_indoor.txt')
    trainLoader = torch.utils.data.DataLoader(trainFolder, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)

    for i, (event_vox, img, mask, gt_img) in enumerate(trainLoader):
        print(i, mask.sum())

    # prefetcher = data_prefetcher(trainLoader)
    # event_vox, img, mask, gt_img = prefetcher.next()
    # while gt_img is not None:
    #     event_vox, img, mask, gt_img = prefetcher.next()
    #     print(mask.sum())
