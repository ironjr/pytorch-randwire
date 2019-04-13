import torch
import torch.nn.functional as F


def CEWithLabelSmoothingLoss(input, target, eps=0.1):
    '''Calculate cross entropy loss, apply label smoothing if needed
    '''
    target = target.contiguous().view(-1)

    if eps is not 0:
        nclass = input.size(1)

        # one hot encoding
        one_hot = torch.zeros_like(input).scatter_(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (nclass - 1)
        log_prob = F.log_softmax(input, dim=1)

        loss = -(one_hot * log_prob).sum(dim=1)
        loss = loss.mean()
    else:
        loss = F.cross_entropy(input, target)

    return loss
