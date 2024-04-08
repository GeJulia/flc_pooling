'''ASAP module
can be used and distributed under the MIT license
Reference:
[1] Grabinski, J., Jung, S., Keuper, J., & Keuper, M. (2022). 
    "FrequencyLowCut Pooling--Plug & Play against Catastrophic Overfitting."
    European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
[2] Grabinski, J., Keuper, J. and Keuper, M. 
    "Fix your downsampling ASAP! Be natively more robust via Aliasing and Spectral Artifact free Pooling." 
    arXiv preprint arXiv:2307.09804 (2023).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ASAP(nn.Module):
    # pooling trough selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    # using a hamming window to prevent sinc-interpolation artifacts
    def __init__(self, transpose=True):
        self.transpose = transpose
        self.window2d = None
        super(ASAP, self).__init__()

    def forward(self, x):
        
        if self.window2d is None:
            window1d = np.abs(np.hamming(x.size(2)))
            window2d = np.sqrt(np.outer(window1d,window1d))
            self.window2d = torch.Tensor(window2d).cuda()
            del window1d
            del window2d
            
        if self.transpose:
            x = x.transpose(2,3)
        low_part = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        low_part = low_part*self.window2d.unsqueeze(0).unsqueeze(0)
        low_part = low_part[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)]
        
        return torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward').real


class ASAP_padding_one(nn.Module):
    # pooling trough selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    # using a hamming window to prevent sinc-interpolation artifacts
    def __init__(self):
        self.window2d = None
        super(ASAP_padding_one, self).__init__()

    def forward(self, x):
        
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        if not torch.is_tensor(self.window2d):
            window1d = np.abs(np.hamming(x.size(2)))
            window2d = np.sqrt(np.outer(window1d,window1d))
            self.window2d = torch.Tensor(window2d).cuda()
            del window1d
            del window2d
            
        low_part = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        low_part = low_part*self.window2d.unsqueeze(0).unsqueeze(0)
        low_part = low_part[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)]
        
        fc = torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward').real
        return fc


class ASAP_padding_large(nn.Module):
    # pooling trough selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    # using a hamming window to prevent sinc-interpolation artifacts
    def __init__(self):
        self.window2d = None
        super(ASAP_padding_large, self).__init__()

    def forward(self, x):
        
        x = F.pad(x, (int(x.size(2)/2-1), int(x.size(2)/2), int(x.size(3)/2-1), int(x.size(3)/2)), "constant", 0)
        if not torch.is_tensor(self.window2d):
            window1d = np.abs(np.hamming(x.size(2)))
            window2d = np.sqrt(np.outer(window1d,window1d))
            self.window2d = torch.Tensor(window2d).cuda()
            del window1d
            del window2d
            
        low_part = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        low_part = low_part*self.window2d.unsqueeze(0).unsqueeze(0)
        low_part = low_part[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)]
        
        fc = torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward').real
        fc = fc[:,:,int( fc.size(2)/4):int(3*fc.size(2)/4),int(fc.size(3)/4): int(3*fc.size(3)/4)]
        return fc
