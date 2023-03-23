# coding=utf-8
"""
support the length of IterDataPipe dataset in distibuted env
"""
from torch.utils.data import DataLoader, Sampler, Dataset, _utils
import torch.distributed as dist
from typing import Any, Optional, Union, TypeVar, Iterable, Sequence, Callable, List
import math

class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]

class DataLoader(DataLoader):
    """
    由于DataLoader的问题，Iterable类型的dataset无法处理分布式情况下Dataloader真实长度的问题，
    len(loader)=每个replicas的sample_num*world_size，这里继承重写Dataloader的__len__方法
    """
    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: Optional[int] = 2,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, 
                collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, 
                generator, 
                prefetch_factor=prefetch_factor, 
                persistent_workers=persistent_workers, 
                pin_memory_device=pin_memory_device)
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = 1

    def __len__(self) -> int:
        """
        just uesed to evaluate the number of batch sample, the result can be used to compute warmup steps
        """
        if self._dataset_kind == _DatasetKind.Iterable:
            length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore[assignment, arg-type]
            if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
                length = math.ceil(
                    (length // self.batch_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
                )
            else:
                length = math.ceil(length / (self.batch_size * self.num_replicas)) 
            return length
        else:
            return len(self._index_sampler)