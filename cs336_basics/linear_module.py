import math
import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.W = torch.empty(out_features, in_features, device=device,dtype = dtype)
        self.W = nn.Parameter(self.W)
        std = math.sqrt(2.0 / (in_features + out_features))
        init.trunc_normal_(self.W, mean=0.0, std=std, a= -3*std,b= 3*std )
    
    def forward(self,x):
        return x @ self.W.T
        