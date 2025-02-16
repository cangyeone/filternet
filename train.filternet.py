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
        self.layer2 = Conv2d(base*2, base*2, kernel, stride, padding=pad)
        self.layer3 = Conv2d(base*2, base*4, kernel, stride, padding=pad) 
        self.layer4 = Conv2d(base*4, base*4, kernel, stride, padding=pad) 
        self.layer5 = Conv2d(base*4, base*8, kernel, [2, 1], padding=pad) 

        self.rnn = nn.GRU(base*8, base*8, 2, batch_first=True, bidirectional=False)


        self.layer6 = Conv2dT(base*8, base*4, kernel, [2, 1], padding=pad)
        self.layer7 = Conv2dT(base*4, base*4, kernel, stride, padding=pad)
        self.layer8 = Conv2dT(base*4, base*2, kernel, stride, padding=pad)
        self.layer9 = Conv2dT(base*2, base*2, kernel, stride, padding=pad)
        self.layer10 = Conv2dT(base*2, base, kernel, stride, padding=pad)
        self.layer11 = nn.Conv2d(base, 1, kernel, [1, 1], padding=pad)
    def forward(self, x):
        B, C, T, W = x.shape 
        #print(x.shape)
        x = self.inputs(x)
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4) 
        x6 = self.layer5(x5)
        x6 = x6.squeeze(2)
        x6 = x6.permute(0, 2, 1)
        h = torch.zeros([2, B, 64], dtype=x6.dtype, device=x6.device)
        x6, h = self.rnn(x6, h)
        x6 = x6.permute(0, 2, 1)
        x6 = x6.unsqueeze(2) 
        
        x7 = self.layer6(x6)
        x8 = self.layer7(x7)
        x9 = self.layer8(x8)
        x10 = self.layer9(x9)
        x11 = self.layer10(x10)
        x12 = self.layer11(x11)
        x12 = x12.squeeze(dim=3)
        x12 = x12.sigmoid()
        return x12

class Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lossfn = nn.MSELoss(reduction="sum") 
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

    version = "01"
    model_name = f"ckpt/filternet.pt" #保存和加载神经网络模型权重的文件路径
    device = torch.device("cuda:0")
    model = UNet()
    try:
        model.load_state_dict(torch.load(model_name, map_location=device))
    except:
        print("模型不存在！")
        pass 
    #for key, var in model.named_parameters():
    #    if var.dtype != torch.float32:continue # BN统计计数无梯度
    #    if "decoder_event_type" in key: # 仅有最后一层有out
    #        var.requires_grad = True
    #    else:
    #        var.requires_grad = False  
    model.to(device)
    model.train()
    lossfn = Loss() 
    lossfn.to(device)
    acc_time = 0 #记录训练的累计时间
    outloss = open(f"logdir/new.filter.rnn2.txt", "a") #记录训练过程中的loss
    optim = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=0e-3)
    fig = plt.figure(1, figsize=(16, 12))
    gs = grid.GridSpec(3, 1)
    count = 0 
    for step in range(50000001):
        for xs, ds in train_dataloader:
            #print(xs.shape, ds.shape)
            x = xs.to(device) 
            d = ds.to(device)
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
                x = x.detach().cpu().numpy() 
                d = d.detach().cpu().numpy() 
                y = y.detach().cpu().numpy() 
                outs = [x, d, y]
                names = ["Input", "Label", "Output"]
                for idx in range(3): 
                    ax = fig.add_subplot(gs[idx]) 
                    ax.matshow(outs[idx][0, 0], cmap=plt.get_cmap("Greys"))
                    ax.set_ylabel(names[idx], fontsize=18)
                plt.savefig("logdir/demo.2.png")
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

