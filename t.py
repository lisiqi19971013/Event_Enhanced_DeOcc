import torch

a = torch.load('./log/checkpoint.pth')
a1 = {'state_dict':a['state_dict']}
torch.save(a1, './log/ckpt.pth')