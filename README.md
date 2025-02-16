### FilterNet: A CNN-RNN based filter model used for raw tunnel lining GPR data
 Ground-Penetrating Radar (GPR) technology, with its characteristics of being fast, non-destructive, and high-resolution, has become an important tool for underground structure detection. However, GPR data inevitably suffer from environmental noise and electromagnetic interference during the acquisition process, leading to decreased data quality and increased complexity in data processing. Traditional filtering algorithms have limitations such as low discrimination between noise and signal, poor adaptability, and inability to process data in real time. This paper proposes a filtering model based on deep neural networks, called FilterNet. FilterNet combines CNN and recurrent neural networks (RNN) for processing multi-channel data. It can perform end-to-end filtering directly on the raw tunnel lining GPR data, achieving functions such as removing air reflection waves, denoising, and automatic gain. Using PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) as statistical indicators, it is shown that the FilterNet model improves filtering precision. The SSIM of all three models is 0.997, and the PSNR of FilterNet1D and FilterNet are 19.06 and 19.41, respectively. Furthermore, tests on the model's processing efficiency indicate that FilterNet requires less memory and is more efficient than the UNet model. FilterNet's parameters are only 48% of those of UNet, its GFLOPS (Giga Floating Point Operations Per Second) is only one-third of UNet's, and it can process data in real time. Additionally, FilterNet performs exceptionally well in suppressing random noise. 

### Usage 
You can use train.xx.py to train the models. 
and use the mkjit.xx.py to generate the inference model. 
you can directly load the model by：

'''Python
import numpy as np 
import torch 

device = torch.device("cuda")
filter = torch.jit.load("ckpt/filternet.jit") 
filter.eval()

# fake data
orignal_data = torch.randn([512, 1024])
with torch.no_grad():
    filtered_data = filter(orignal_data)
'''