import torch
import torch.nn as nn
from cs336_basics.linear_module import Linear

class SwiGLU_FFN(nn.Module):

    def __init__(self,d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_up= d_ff
        self.W1 = Linear(self.d_model, self.d_up, device=device, dtype=dtype)  
        self.W2 = Linear(self.d_up, self.d_model, device=device, dtype=dtype)  
        self.W3 = Linear(self.d_model, self.d_up, device=device, dtype=dtype)  
    
    def forward(self,x):
        u = self.W1(x)
        v = self.W3(x) #gate
        u = u * torch.sigmoid(u) # SiLU
        a = u * v
        final = self.W2(a)
        return final

