import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.spatial.distance import cdist


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
    def __init__(self, alpha: float, beta: float, threshold: float = .5, 
                 reduction: str = 'sum', smooth = 1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.reduction = reduction
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        assert targets.size() == inputs.size()
        for i in range(len(targets)):
            tp = (targets[i].flatten() * inputs[i].flatten()).sum()
            fn = targets[i].sum() - tp 
            fp = inputs[i].sum() - tp
            tversky = (tp + self.smooth)  / (tp + self.alpha * fn + self.beta * fp + self.smooth)
            tversky_loss = 1 - tversky
            if i == 0:
                loss = tversky_loss
            else:
                loss += tversky_loss
            
        return loss
    
class CustomDiceLoss(nn.Module):
    """
    Implementation of Dice Loss using the boarder weights from the original U-Net paper.
    """
    def __init__(self, smooth: float = 1.0, reduction='sum', sigma: int = 5, weight_bias: int = 10):
        super(CustomDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.sigma = sigma
        self.weight_bias = weight_bias
        
    def _get_boarder_weights(self, true_mask):
        """
        This function takes the true mask as input and returns for each class in a binary mask setting
        the minimal euclidean distance from one pixel segment to the other segment
        
        """
        # extract masked cell
        mask = torch.clone(true_mask.detach().cpu())
        if mask.sum() > 1:
            #print('Sum',mask.sum())
            #print('Shape', mask.shape)
            mask_pix = (mask == 1).nonzero().cpu().numpy()[:, 1:]
            foreground_pix = (mask == 0).nonzero().cpu().numpy()[:, 1:]
            assert mask_pix.ndim == 2, f'Dim not correct {mask_pix.ndim}'
            
            # calculate dist matrix between each element in mask and foreground
            dist_mat = cdist(mask_pix, foreground_pix)

            # extract minium nearest boarder for each pixel in mask
            #print(dist_mat.shape)
            nearest_from_segment = np.min(dist_mat, axis=1)
            nearest_from_foreground = np.min(dist_mat, axis=0)
            # applying formula from unet paper
            nearest_from_segment = self.weight_bias * np.exp(-1* (nearest_from_segment / (2*self.sigma**2)))
            nearest_from_foreground = self.weight_bias * np.exp(-1* (nearest_from_foreground / (2*self.sigma**2)))
            
            # assign to mask elements as a new weight tensor
            mask[(mask == 1)] = torch.Tensor(nearest_from_segment)
            mask[(mask == 0)] = torch.Tensor(nearest_from_foreground)
        else:
            # if there is only one class then the mask consists only of ones.
            mask = torch.ones(true_mask.shape)
        
        return mask

    def forward(self, inputs, targets):
        assert inputs.size() == targets.size(), f'{inputs.size}, {targets.size()}'
        for i in range(len(targets)):
            # get weights for true
            boarder_weights = self._get_boarder_weights(targets[i].detach()).contiguous().view(-1)
            boarder_weights = boarder_weights.to('cuda' if torch.cuda.is_available() else 'cpu')
            y_pred = inputs[i].contiguous().view(-1)
            y_true = targets[i].contiguous().view(-1)
            
            numerator = 2. * (boarder_weights * (y_pred * y_true)).sum() + self.smooth
            denominator = (boarder_weights * (y_pred + y_true)).sum() + self.smooth
            dsc = 1. - (numerator / denominator)
            if i == 0:
                loss = dsc
            else:
                loss += dsc
                
        if self.reduction == 'mean':
            loss /= len(y_true)
            
        return loss