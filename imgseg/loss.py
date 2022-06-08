import torch
import numpy as np
import pandas as pd
import torch.nn as nn



class DiceLoss(nn.Module):
    """
    https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/d45f8908ab2f0246ba204c702a6161c9eb25f902/loss.py#L4
    """
    def __init__(self, smooth: float = 1.0, reduction='sum'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        assert inputs.size() == targets.size()
        for i in range(len(targets)):
            y_pred_ = inputs[i].contiguous().view(-1)
            y_true_ = targets[i].contiguous().view(-1)
            intersection = (y_pred_ * y_true_).sum()
            dsc = (2. * intersection + self.smooth) / (
                y_pred_.sum() + y_true_.sum() + self.smooth
            )
            dsc_loss = 1. - dsc
            if i == 0:
                loss = dsc_loss
            else:
                loss += dsc_loss
        if self.reduction == 'mean':
            loss /= len(y_true)
            
        return loss
    
    
class InverseWeightedDiceLoss(nn.Module):
    """
    Implementation according to paper: https://pubmed.ncbi.nlm.nih.gov/34442075/
    """
    def __init__(self, per_class_frequencies: int, exp: float = 1., reduction: str = 'sum', smooth: int = 1):
        super(InverseWeightedDiceLoss, self).__init__()
        self.per_class_frequencies = per_class_frequencies
        self.inverse_frequencies = self._init_inverse_weights(per_class_frequencies, exp)
        self.exp = exp
        self.reduction = reduction
        self.smooth = smooth
        
    @staticmethod
    def _init_inverse_weights(per_class_frequencies: list, exp: float):
        return 1 / (np.array(per_class_frequencies) ** exp)
    
    def forward(self, inputs, targets):
        assert inputs.size() == targets.size()
        for i in range(len(targets)):
            # Assign binary weights according to ground truth
            weight_mask = torch.where(targets[i] == 1, 
                                      self.inverse_frequencies[1], 
                                      self.inverse_frequencies[0]).flatten()
            # Build mask bc one out layer for class labels
            intersection = (weight_mask * (inputs[i].flatten() * targets[i].flatten())).sum()
            union = (weight_mask * (inputs[i].flatten() + targets[i].flatten())).sum()
            dsc = (2. * intersection + self.smooth) / (union + self.smooth)
            score = 1 - dsc
            
            # Peform reduction
            if i == 0:
                loss = score
            else:
                loss += score 
                    
        if self.reduction == 'mean':
            loss /= len(targets)
            
        return loss

class TverskyLoss(nn.Module):
    """Tversky Loss only for binary labels."""
    def __init__(self, alpha: float, beta: float, threshold: float = .5, reduction: str = 'sum', smooth = 1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.reduction = reduction
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        assert targets.size() == inputs.size()
        losses = []
        for i in range(len(targets)):
            tp = (targets[i].flatten() * inputs[i].flatten()).sum()
            fn = targets[i].sum() - tp 
            fp = torch.where(inputs[i] >= self.threshold, 1, 0).sum() - tp
            tversky = (tp + self.smooth)  / (tp + self.alpha * fn + self.beta * fp + self.smooth)
            losses.append(1-tversky)
            
        # Perform reduction
        losses = torch.Tensor(losses)
        if self.reduction == 'sum':
            loss = torch.sum(losses)
        elif self.reduction == 'mean':
            loss = torch.mean(losses)
            
        return loss