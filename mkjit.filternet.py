from colorsys import yiq_to_rgb
import torch 
import torch.nn as nn 
import time 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
import torch.nn.utils.rnn as rnnutils
import torch 
import numpy as np 
import os 
class MyDataset2(Dataset):
    def __init__(self):
        root = "data/datas" 
        file_names = os.listdir(root)
        self.datas = []
        for fn in file_names:
            path = os.path.join(root, fn) 
            self.datas.append(path)
            #f = np.load(path) 

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        N = len(self.datas)
        path = self.datas[idx%N] 
        f = np.load(path) 
        data = f["data"] 
        label = f["label"]
        C, T = data.shape 
        idx = np.random.randint(0, T-2048)
        sample_x = data[:, idx:idx+2048]
        sample_d = label[:, idx:idx+2048]

        sample_x = sample_x.astype(np.float32) 
        sample_d = sample_d.astype(np.float32)

        sample_x -= np.mean(sample_x, axis=0, keepdims=True) 

        sample_x -= np.min(sample_x) 
        sample_x /= np.max(sample_x) + 1e-6 

        sample_d -= np.min(sample_d) 
        sample_d /= np.max(sample_d) + 1e-6 


        sample_x = torch.tensor(sample_x, dtype=torch.float32) 
        sample_d = torch.tensor(sample_d, dtype=torch.float32) 
        return sample_x, sample_d 


class MyDataset(Dataset):
    def __init__(self):
        root = "data/segs" 
        file_names = os.listdir(root)
        self.datas = []
        for fn in file_names:
            path = os.path.join(root, fn) 
            self.datas.append(path)
            #f = np.load(path) 

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        N = len(self.datas)
        path = self.datas[idx%N] 
        f = np.load(path) 
        data = f["data"] 
        label = f["label"]
        sample_x = data
        sample_d = label

        sample_x = sample_x.astype(np.float32) 
        sample_d = sample_d.astype(np.float32)

        #sample_x -= np.mean(sample_x, axis=1, keepdims=True) 

        sample_x -= np.min(sample_x) 
        sample_x /= np.max(sample_x) + 1e-6 




        #sample_d -= np.min(sample_d) 
        #sample_d /= np.max(sample_d) + 1e-6 
        mean = np.mean(sample_d)
        std = np.std(sample_d)
        ll = mean - (std * 3) # lower color limit
        ul = mean + (std * 3) # upper color limit
        imgdata = np.clip(sample_d, ll, ul)
        sample_d = (imgdata - np.min(imgdata)) / (np.max(imgdata) - np.min(imgdata))

        sample_x = torch.tensor(sample_x, dtype=torch.float32) 
        sample_d = torch.tensor(sample_d, dtype=torch.float32) 
        return sample_x, sample_d 

def collate_batch(batch):
    """
    定义后处理函数
    """
    xs, ds = [], []
    for x, d in batch:
        xs.append(x) 
        ds.append(d)
    xs = torch.stack(xs, dim=0).unsqueeze(dim=1) 
    ds = torch.stack(ds, dim=0).unsqueeze(dim=1) 
    return xs, ds 

import torch  
import torch.nn as nn 
import torch.nn.functional as F 
class Conv2d(nn.Module):
    def __init__(self, nin=8, nout=11, ks=[7, 1], st=[4, 1], padding=[3, 0]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(nin, nout, ks, st, padding=padding), 
            nn.BatchNorm2d(nout), 
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class Conv2dT(nn.Module):
    def __init__(self, nin=8, nout=11, ks=[7, 1], st=[4, 1], padding=[3, 0]):
        super().__init__()
        # 这里我们使用上采样进行
        self.layers = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=tuple(st)), 
            Conv2d(nin, nout, ks, [1, 1], padding=padding), 
            nn.BatchNorm2d(nout), 
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class FilterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = Conv2d(1, 8, [7, 1], [1, 1], padding=[3, 0]) 
        self.layer0 = Conv2d(8, 8, [7, 1], [1, 1], padding=[3, 0]) 
        self.layer1 = Conv2d(8, 16, [7, 1], [4, 1], padding=[3, 0])
        self.layer2 = Conv2d(16, 16, [7, 1], [4, 1], padding=[3, 0])
        self.layer3 = Conv2d(16, 32, [7, 1], [4, 1], padding=[3, 0]) 
        self.layer4 = Conv2d(32, 32, [7, 1], [4, 1], padding=[3, 0]) 
        self.layer5 = Conv2d(32, 64, [7, 1], [2, 1], padding=[3, 0]) 

        self.rnn = nn.GRU(64, 64, 2, batch_first=True, bidirectional=False)


        self.layer6 = Conv2dT(64, 32, [7, 1], [2, 1], padding=[3, 0])
        self.layer7 = Conv2dT(32, 32, [7, 1], [4, 1], padding=[3, 0])
        self.layer8 = Conv2dT(32, 16, [7, 1], [4, 1], padding=[3, 0])
        self.layer9 = Conv2dT(16, 16, [7, 1], [4, 1], padding=[3, 0])
        self.layer10 = Conv2dT(16, 8, [7, 1], [4, 1], padding=[3, 0])
        self.layer11 = nn.Conv2d(8, 1, [7, 1], [1, 1], padding=[3, 0])
    def forward(self, x):
        x = x.squeeze() 
        x -= torch.min(x) 
        x /= torch.max(x) + 1e-6 
        x = x.unsqueeze(0) 
        x = x.unsqueeze(1) 
        x = self.inputs(x)
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4) 
        x6 = self.layer5(x5)
        #print(x6.shape)
        x6 = x6.squeeze(2)
        x6 = x6.permute(0, 2, 1)
        h = torch.zeros([2, 1, 64], dtype=x6.dtype, device=x6.device)
        x6, h = self.rnn(x6, h)
        #print(x6.shape)
        x6 = x6.permute(0, 2, 1)
        x6 = x6.unsqueeze(2) 
        
        x7 = self.layer6(x6)
        x8 = self.layer7(x7)
        x9 = self.layer8(x8)
        x10 = self.layer9(x9)
        x11 = self.layer10(x10)
        x12 = self.layer11(x11)
        #x10 = F.softmax(x10, dim=1)
        #print(x12.shape)
        x12 = x12.squeeze(dim=3)
        x12 = x12.sigmoid()
        x12 = x12.squeeze()
        return x12
def main(args):
    model_name = f"ckpt/filternet.pt" #保存和加载神经网络模型权重的文件路径
    device = torch.device("cpu")
    model = FilterNet()
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()
    model(torch.randn([512, 2133]))
    from thop import profile, clever_format
    x = torch.randn([512, 2048])
    flops, params = profile(model, inputs=(x, ))
    print(clever_format([flops, params], "%.3f"))
    torch.jit.save(torch.jit.script(model), f"ckpt/filternet.jit")
 
import argparse
if __name__ == "__main__":
         
    main([])

