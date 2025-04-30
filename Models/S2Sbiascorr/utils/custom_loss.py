import torch
import torch.nn as nn
import math


class CRPSLoss(nn.Module):
    def __init__(self):
        super(CRPSLoss, self).__init__()

    def forward(self, mu, sigma, target):
        
        standardized = (target - mu) / sigma

        pdf = torch.exp(-0.5 * standardized**2) / math.sqrt(2 * math.pi)
        cdf = 0.5 * (1 + torch.erf(standardized / math.sqrt(2)))
        
        crps = sigma * (standardized * ((2 * cdf) - 1) + (2 * pdf) - (1 / math.sqrt(math.pi)))
        return crps.mean()


class CRPSMSELoss(nn.Module):
    def __init__(self):
        super(CRPSMSELoss, self).__init__()

    def forward(self, mu, sigma, target):
        
        standardized = (target - mu) / sigma

        pdf = torch.exp(-0.5 * standardized**2) / math.sqrt(2 * math.pi)
        cdf = 0.5 * (1 + torch.erf(standardized / math.sqrt(2)))
        
        crps = sigma * (standardized * ((2 * cdf) - 1) + (2 * pdf) - (1 / math.sqrt(math.pi)))

        mse = torch.mean((mu-target)**2)
        
        return crps.mean() + 0.1*mse

