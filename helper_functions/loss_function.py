import torch.nn as nn
import torch

def loss_function(output, target, criterion = nn.MSELoss().cuda()):

    idx             = ~torch.isnan(target)
    output_final    = output[idx]
    target_final    = target[idx]
    print(idx)
    print(output_final.shape, target_final.shape)

    loss            = criterion(output_final, target_final)

    return loss

