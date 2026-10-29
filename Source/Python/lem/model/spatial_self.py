# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Let local self forecasts inspect spatial sense features before pooling."""
import torch
from torch import nn


class SpatialSelfContext(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.query=nn.Linear(width,width,bias=False)
        self.key=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width,bias=False)
        self.output=nn.Linear(width,width,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,local,spatial,valid):
        if spatial.ndim!=3 or spatial.shape[0]!=local.shape[0] or valid.shape!=spatial.shape[:2]:
            raise ValueError('spatial self context lost body or region identity')
        if valid.dtype!=torch.bool or not torch.isfinite(spatial).all():
            raise ValueError('invalid spatial self observation')
        if not spatial.shape[1]:return torch.zeros_like(local)
        available=valid.any(-1)
        scores=torch.einsum('bqd,bkd->bqk',self.query(local),self.key(spatial))/local.shape[-1]**.5
        scores=scores.masked_fill(~valid[:,None],-torch.inf)
        # Missing real vision is not an imagined image. Its contribution is zero.
        scores=torch.where(available[:,None,None],scores,torch.zeros_like(scores))
        context=torch.einsum('bqk,bkd->bqd',scores.softmax(-1),self.value(spatial))
        return self.output(context)*available[:,None,None]
