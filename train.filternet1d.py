import torch 
import torch.nn as nn 
import time 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
import torch.nn.utils.rnn as rnnutils
import torch 
import numpy as np 
import os 


class MyDataset(Dataset):
    def __init__(self):
        root = "data/segs" 
        file_names = os.listdir(root)
        self.datas = []
        for fn in file_names:
            if ".0." in fn or ".1." in fn:continue 
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


        mean = np.mean(sample_d)
        std = np.std(sample_d)
        ll = mean - (std * 3) # lower color limit
        ul = mean + (std * 3) # upper color limit
        imgdata = np.clip(sample_d, ll, ul)
        sample_d = (imgdata - np.min(imgdata)) / (np.max(imgdata) - np.min(imgdata))
        #sample_d -= np.mean(sample_d) 
        #sample_d /= np.max(sample_d) + 1e-6 


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
    xs = torch.stack(xs, dim=0)#.unsqueeze(dim=1) 
    ds = torch.stack(ds, dim=0)#.unsqueeze(dim=1) 
    return xs, ds 

import torch  
import torch.nn as nn 
import torch.nn.functional as F 
class Conv2d(nn.Module):
    def __init__(self, nin=8, nout=11, ks=kernel, st=stride, padding=pad):
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
    def __init__(self, nin=8, nout=11, ks=kernel, st=stride, padding=pad):
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
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = 8 
        kernel = [7, 1]
        stride = [4, 1]
        pad = [3, 0]
        self.inputs = Conv2d(1, base, kernel, [1, 1], padding=pad) 
        self.layer0 = Conv2d(base, base, kernel, [1, 1], padding=pad) 
        self.layer1 = Conv2d(base, base*2, kernel, stride, padding=pad)
        self.layer2 = Conv2d(base*2, base*4, kernel, stride, padding=pad)
        self.layer3 = Conv2d(base*4, base*8, kernel, stride, padding=pad) 
        self.layer4 = Conv2d(base*8, base*16, kernel, stride, padding=pad) 
        self.layer5 = Conv2dT(base*16, base*8, kernel, stride, padding=pad)
        self.layer6 = Conv2dT(base*16, base*4, kernel, stride, padding=pad)
        self.layer7 = Conv2dT(base*8, base*2, kernel, stride, padding=pad)
        self.layer8 = Conv2dT(base*4, base, kernel, stride, padding=pad)
        self.layer9 = nn.Conv2d(base*2, 1, kernel, [1, 1], padding=pad)
    def forward(self, x):
        x = x.unsqueeze(3)
        x = self.inputs(x)
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4) 
        x6 = self.layer5(x5)
        x6 = torch.cat([x4, x6], dim=1)
        x7 = self.layer6(x6)
        x7 = torch.cat([x3, x7], dim=1)
        x8 = self.layer7(x7)
        x8 = torch.cat([x2, x8], dim=1)
        x9 = self.layer8(x8)
        x9 = torch.cat([x1, x9], dim=1)
        x10 = self.layer9(x9)
        x10 = x10.squeeze(dim=3)
        x10 = x10.sigmoid()
        return x10

class Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lossfn = nn.L1Loss(reduction="sum") 
    def forward(self, x, d):
        loss = self.lossfn(x, d) 
        #loss = - (d * torch.log(x+1e-6)).sum()
        return loss 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as grid 
def main(args):

    train_dataset = MyDataset()     
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, 
        shuffle=True, collate_fn=collate_batch, num_workers=3)

    version = "03x"
    model_name = f"ckpt/filternet1d.pt" #保存和加载神经网络模型权重的文件路径
    device = torch.device("cuda")
    model = UNet()
    try:
        model.load_state_dict(torch.load(model_name, map_location=device))
    except:
        print("模型不存在！")
        pass 
    model.to(device)
    model.train()
    lossfn = Loss() 
    lossfn.to(device)
    acc_time = 0 #记录训练的累计时间
    outloss = open(f"logdir/new.filter.1d.txt", "a") #记录训练过程中的loss
    optim = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=0e-3)
    fig = plt.figure(1, figsize=(16, 12))
    gs = grid.GridSpec(3, 1)
    count = 0 
    for step in range(50000001):
        for xs, ds in train_dataloader:
            #print(xs.shape, ds.shape)
            xs = xs.to(device) 
            ds = ds.to(device)
            xs = xs.permute(0, 2, 1)
            ds = ds.permute(0, 2, 1)
            for i in range(32):
                x = xs[0].unsqueeze(1) 
                d = ds[0].unsqueeze(1)
                #x = torch.tensor(xs, dtype=torch.float32, device=device) 
                #d = torch.tensor(ds, dtype=torch.float32, device=device)
                #print(x.shape, d.shape)
                y = model(x)
                loss = lossfn(y, d) 
                loss.backward()
                if loss.isnan():
                    print("NAN error")
                    optim.zero_grad()
                    continue 
                optim.step() 
                optim.zero_grad()
            ls = loss.detach().cpu().numpy()
            count += 1
            if count % 10 == 0:
                x = x.detach().squeeze().cpu().numpy() 
                d = d.detach().squeeze().cpu().numpy() 
                y = y.detach().squeeze().cpu().numpy() 
                outs = [x, d, y]
                names = ["Input", "Label", "Output"]
                for idx in range(3): 
                    ax = fig.add_subplot(gs[idx]) 
                    ax.matshow(outs[idx].T, cmap=plt.get_cmap("Greys"))
                    ax.set_ylabel(names[idx], fontsize=18)
                plt.savefig("logdir/demo.1d.png")
                plt.cla() 
                plt.clf()
                torch.save(model.state_dict(), model_name)
                print(f"{step},{ls}\n")
                outloss.write(f"{step},{ls}\n")
                outloss.flush()
    print("done!")
    print("done!")
#nohup python china.large6.py > logdir/large6.log 2>&1 &
#3583746
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-d', '--dist', default=200, type=int, help="输入连续波形")       
    parser.add_argument('-o', '--output', default="result/t1", help="输出文件名")      
    parser.add_argument('-m', '--model', default="lppn.model", help="模型文件lppnmodel")                                                            
    args = parser.parse_args()      
    main(args)

