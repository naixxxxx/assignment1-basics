import torch
import torch.nn as nn
import torch.nn.init as init

class Embedding(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embeddings = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.embeddings = nn.Parameter(self.embeddings)
        init.trunc_normal_(self.embeddings, mean=0.0, std=1.0, a=-3, b=3)
    
    def forward(self,token_ids: torch.Tensor):
        return self.embeddings[token_ids]
