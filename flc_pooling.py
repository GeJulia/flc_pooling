'''FLC Pooling module
can be used and distributed under the MIT license'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FLC_Pooling(nn.Module):
    # pooling torugh selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self):
        super(FLC_Pooling, self).__init__()

    def forward(self, x):

        low_part = torch.fft.fftshift(torch.fft.fft2(x))[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)]
        
        return abs(torch.fft.ifft2(torch.fft.ifftshift(low_part)))