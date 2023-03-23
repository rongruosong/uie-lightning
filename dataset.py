# coding=utf-8
"""
adapted from paddlenlp
"""
import torchdata
from torchdata.datapipes import functional_datapipe
# from torch.utils.data import DataLoader
from dataloader import DataLoader
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper, FileOpener

from typing import IO, Tuple, Dict, Optional, Callable, Iterator, List, Any
import json
from transformers import BertTokenizerFast
import torch
import time


@functional_datapipe("parse_line_json_files")
class LineJsonParserIterDataPipe(IterDataPipe[Tuple[str, Dict]]):
    r"""
    加载解析分行的json文件(functional name: ``parse_line_json_files``).
    Args:
        source_datapipe: a DataPipe with tuples of file name and JSON data stream
        kwargs: keyword arguments that will be passed through to ``json.loads``
    """

    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, IO]], **kwargs) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, IO]] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Dict]:
        for file_name, stream in self.source_datapipe:
            for line in stream:
                yield json.loads(line, **self.kwargs)


@functional_datapipe("truncation_line")
class LineTruncationIterDataPipe(IterDataPipe):
    r"""
    处理每行数据截取最大长度
    """
    def __init__(self, source_datapipe: IterDataPipe[Dict[str, Any]], max_seq_len: int):
        self.source_datapipe = source_datapipe
        self.max_seq_len = max_seq_len
    
    def __iter__(self):
        for json_line in self.source_datapipe:
            content = json_line['content'].strip()
            prompt = json_line['prompt']
            # Model Input is aslike: [CLS] Prompt [SEP] Content [SEP]
            # It include three summary tokens.
            if self.max_seq_len <= len(prompt) + 3:
                raise ValueError(
                    "The value of max_seq_len is too small, please set a larger value"
                )
            max_content_len = self.max_seq_len - len(prompt) - 3
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line['result_list']
                json_lines = []
                accumulate = 0
                while True:
                    cur_result_list = []

                    for result in result_list:
                        if result['start'] + 1 <= max_content_len < result[
                                'end']:
                            max_content_len = result['start']
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]

                    while True:
                        if len(result_list) == 0:
                            break
                        elif result_list[0]['end'] <= max_content_len:
                            if result_list[0]['end'] > 0:
                                cur_result = result_list.pop(0)
                                cur_result_list.append(cur_result)
                            else:
                                cur_result_list = [
                                    result for result in result_list
                                ]
                                break
                        else:
                            break

                    json_line = {
                        'content': cur_content,
                        'result_list': cur_result_list,
                        'prompt': prompt
                    }
                    json_lines.append(json_line)

                    for result in result_list:
                        if result['end'] <= 0:
                            break
                        result['start'] -= max_content_len
                        result['end'] -= max_content_len
                    accumulate += max_content_len
                    max_content_len =self.max_seq_len - len(prompt) - 3
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        json_line = {
                            'content': res_content,
                            'result_list': result_list,
                            'prompt': prompt
                        }
                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content

                for json_line in json_lines:
                    yield json_line


@functional_datapipe('convert_example')
class ConvertIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe[Dict[str, Any]], tokenizer: BertTokenizerFast, max_seq_len: int):
        self.source_datapipe = source_datapipe
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
    
    def __iter__(self):
        for example in self.source_datapipe:
            # print(example)
            encoded_inputs = self.tokenizer.encode_plus(text=example["prompt"],
                                                        text_pair=example["content"],
                                                        padding='max_length',
                                                        truncation=True,
                                                        max_length=self.max_seq_len,
                                                        return_offsets_mapping=True)
            # encoded_inputs = encoded_inputs
            # print(encoded_inputs)
            offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
            # tokens = self.tokenizer.convert_ids_to_tokens(encoded_inputs["input_ids"])
            # print(tokens)
            # token_map = [[token, mapping] for token, mapping in zip(tokens, offset_mapping)]
            # print(token_map)

            # 转换offset_mapping，生成prompt+content整体的offset_mapping
            bias = 0
            for index in range(1, len(offset_mapping)):
                mapping = offset_mapping[index]
                if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                    bias = offset_mapping[index - 1][1] + 1  # Includes [SEP] token
                if mapping[0] == 0 and mapping[1] == 0:
                    continue
                offset_mapping[index][0] += bias
                offset_mapping[index][1] += bias
            
            # 根据转换后的offset_mapping，处理starts与ends
            start_ids = [0 for x in range(self.max_seq_len)]
            end_ids = [0 for x in range(self.max_seq_len)]
            for item in example["result_list"]:
                start = self.map_offset(item["start"] + bias, offset_mapping)
                end = self.map_offset(item["end"] - 1 + bias, offset_mapping)
                start_ids[start] = 1.0
                end_ids[end] = 1.0

            tokenized_output = [
                encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
                encoded_inputs["attention_mask"],sum(encoded_inputs["attention_mask"]),
                start_ids, end_ids
            ]
            tokenized_output = [torch.tensor(x, dtype=torch.int64) for x in tokenized_output]
            # yield tuple(tokenized_output)
            yield tokenized_output

    def map_offset(self, ori_offset: int, offset_mapping: List[List[int]]) -> int:
        """
        map ori offset to token offset
        """
        for index, span in enumerate(offset_mapping):
            if span[0] <= ori_offset < span[1]:
                return index
        return -1

@functional_datapipe("bucket")
class BucketIterDataPipe(IterDataPipe):
    r"""
    The purpose of this DataPipe is to sort samples with some similarity according to the sorting function
    being passed. 
    """
    datapipe: IterDataPipe
    bucket_size: int
    drop_last: bool
    bucket_num: int
    sort_key: Optional[Callable]
    use_in_batch_shuffle: bool

    def __new__(
        cls,
        datapipe: IterDataPipe,
        bucket_size: int,
        drop_last: bool = False,
        bucket_num: int = 1,
        sort_key: Optional[Callable] = None,
        use_in_batch_shuffle: bool = True,
    ):
        assert bucket_size > 0, "Bucket size is required to be larger than 0!"
        assert bucket_num > 0, "Number of buckets is required to be larger than 0!"

        pool_size = bucket_size * bucket_num

        # Shuffle by pool_size
        if bucket_num > 1 or sort_key is None:
            if use_in_batch_shuffle:
                datapipe = datapipe.batch(batch_size=pool_size, drop_last=False).in_batch_shuffle().unbatch()
            else:
                datapipe = datapipe.shuffle(buffer_size=pool_size)
        # Sort by bucket_size if sort_key is given
        if sort_key is not None:
            datapipe = datapipe.batch(bucket_size).map(fn=sort_key).unbatch()
        return datapipe

def collate_fn(batch: List[torch.Tensor]) -> List[torch.Tensor]:
    input_ids, token_type_ids, attention_mask, lengths, start_ids, end_ids = map(torch.stack, zip(*batch))
    max_len = max(lengths).item()
    output = [input_ids, token_type_ids, attention_mask, start_ids, end_ids]
    return [x[:, :max_len] for x in output]

def sort_bucket(bucket):
    """
    使用token长度对bucket进行排序
    """
    bucket = sorted(bucket, key=lambda x:x[3].item())
    return bucket

def create_data_loader_future(
    data_path: str,
    tokenizer: BertTokenizerFast, 
    max_seq_len: int,
    batch_size: int = 32,
    num_workers: int = 1,
    mode: str = 'train'
):
    datapipe = IterableWrapper([data_path])
    datapipe = datapipe.open_files(encoding="utf-8").parse_line_json_files()
    if mode == 'train':
        datapipe = datapipe.shuffle()
    datapipe = datapipe.truncation_line(max_seq_len=max_seq_len).sharding_filter()
    datapipe = datapipe.convert_example(tokenizer=tokenizer, max_seq_len=max_seq_len)
    datapipe = datapipe.batch(batch_size).collate(collate_fn)

    # set dataloader
    mp_rs = MultiProcessingReadingService(num_workers=num_workers)
    dist_rs = DistributedReadingService()
    rs = SequentialReadingService(dist_rs, mp_rs)

    loader = DataLoader2(datapipe, reading_service=rs)
    return loader

def create_data_loader(
    data_path: str,
    tokenizer: BertTokenizerFast, 
    max_seq_len: int,
    batch_size: int = 32,
    num_workers: int = 1,
    world_size: int = 1,
    bucket_num: int = 1,
    mode: str = 'train',
    use_bucket: bool = False
) -> DataLoader:
    if '0.4' in torchdata.__version__:
        from utils import LengthSetterIterDataPipe
    
    datapipe = IterableWrapper([data_path])
    datapipe = datapipe.open_files(encoding="utf-8").parse_line_json_files()
    if mode == 'train':
        datapipe = datapipe.shuffle()
    datapipe = datapipe.truncation_line(max_seq_len=max_seq_len).sharding_filter()
    length = sum([1 for _ in datapipe])
    datapipe = datapipe.convert_example(tokenizer=tokenizer, max_seq_len=max_seq_len)
    if use_bucket:
        datapipe = datapipe.bucket(bucket_size=batch_size*100, bucket_num=bucket_num, sort_key=sort_bucket)
    # if world_size > 1:
    #     assert num_workers <= 1, "use "
    #     datapipe = datapipe.fullsync()
    datapipe = datapipe.set_length(length)
    
    # set dataloader
    loader = DataLoader(datapipe, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    return loader