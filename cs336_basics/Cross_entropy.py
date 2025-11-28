import torch

def my_logsumexp(x, dim):
    m = x.max(dim=dim, keepdim=True).values
    y = x - m

    m = m.squeeze(dim)
    y = torch.exp(y)
    y = y.sum(dim=dim, keepdim=False)
    y = torch.log(y)
    
    return  m + y

def Cross_entropy(x,target):

    logsumexp = my_logsumexp(x, -1)
    p = x.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    loss = logsumexp - p
    return loss.mean()

"""
    x = softmax(x,-1)  
    p = x.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    loss = -torch.log(p)
    loss = loss.mean()
    return loss

"""