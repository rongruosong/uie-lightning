# coding=utf-8
from typing import Tuple
import torch
from torch import Tensor

def uie_loss_func(
    outputs: Tuple[Tensor, Tensor], 
    labels: Tuple[Tensor, Tensor]
) -> Tensor:
    criterion = torch.nn.BCELoss()
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = start_ids.float()
    end_ids = end_ids.float()
    loss_start = criterion(start_prob, start_ids)
    loss_end = criterion(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    return loss