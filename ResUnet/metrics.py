import torch

def l2_dist(predictions, labels, device):
    a = predictions.clone().detach()
    b = labels.clone().detach()
    dist = torch.dist(a,b)

    return dist.cpu().numpy()

def pearson_correlation(predictions, labels):
    x = predictions.clone().detach()
    y = labels.clone().detach()

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))