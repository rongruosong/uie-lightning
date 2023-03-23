# coding=utf-8
from typing import List, Iterator
import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


def get_ids_greater_than(probs: torch.Tensor, threshold: float = 0.5) -> List[torch.Tensor]:
    """
    Get idx of the last dimension in probability arrays, which is greater than the threshold
    """
    if probs.dim() < 2:
        probs = probs.unsqueeze(0)
    probs_bool = probs.gt(threshold)
    ids_row, ids_col = torch.nonzero(probs_bool, as_tuple=True)
    
    result = []
    shape = probs.size()
    for i in range(shape[0]):
        result.append(torch.masked_select(ids_col, ids_row == i))
    
    return result

def get_span(start_ids, end_ids):
    """
    copy from paddlenlp
    Get span set from position start and end list.
    """
    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        start_id = start_ids[start_pointer]
        end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        if start_id > end_id:
            end_pointer += 1
            continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result

@functional_datapipe("set_length")
class LengthSetterIterDataPipe(IterDataPipe):
    r"""
    Set the length attribute of the DataPipe
    """

    def __init__(self, source_datapipe: IterDataPipe, length: int) -> None:
        self.source_datapipe: IterDataPipe = source_datapipe
        assert length >= 0
        self.length: int = length

    def __iter__(self) -> Iterator:
        yield from self.source_datapipe

    def __len__(self) -> int:
        return self.length