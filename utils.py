import torch

def accuracy(outs, truth):
    preds = outs >= 0.5
    targets = truth >= 0.5
    # Make both predictions and ground truths have the same shape
    preds = torch.reshape(preds, targets.shape)
    return preds.eq(targets).sum().item() / targets.numel()
