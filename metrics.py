# coding=utf-8
"""
adapted from paddlenlp
"""
from typing import Tuple
import torch
from torch import Tensor
from torchmetrics import Metric
from utils import get_ids_greater_than, get_span

class SpanMetric(Metric):
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.add_state("num_infer_spans", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_label_spans", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_correct_spans", default=torch.tensor(0), dist_reduce_fx="sum")
        self.threshold = threshold
    
    def update(
        self, 
        start_probs: Tensor, 
        end_probs: Tensor, 
        gold_start_ids: Tensor, 
        gold_end_ids: Tensor
    ) -> None:
        pred_start_ids = get_ids_greater_than(start_probs, self.threshold)
        pred_end_ids = get_ids_greater_than(end_probs, self.threshold)
        gold_start_ids = get_ids_greater_than(gold_start_ids, self.threshold)
        gold_end_ids = get_ids_greater_than(gold_end_ids, self.threshold)

        for predict_start_ids, predict_end_ids, label_start_ids, label_end_ids in zip(
            pred_start_ids, pred_end_ids, gold_start_ids, gold_end_ids
        ):
            num_correct, num_infer, num_label = self._update_cal(
                predict_start_ids, 
                predict_end_ids, 
                label_start_ids,
                label_end_ids
            )
            
            self.num_infer_spans += num_infer
            self.num_label_spans += num_label
            self.num_correct_spans += num_correct
    
    def _update_cal(
        self, 
        predict_start_ids: Tensor, 
        predict_end_ids: Tensor, 
        label_start_ids: Tensor, 
        label_end_ids: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        evaluate position extraction (start, end)
        return num_correct, num_infer, num_label
        input: [1, 2, 10] [4, 12] [2, 10] [4, 11]
        output: (1, 2, 2)
        """
        pred_set = get_span(predict_start_ids.tolist(), predict_end_ids.tolist())
        label_set = get_span(label_start_ids.tolist(), label_end_ids.tolist())
        num_correct = len(pred_set & label_set)
        num_infer = len(pred_set)
        # For the case of overlapping in the same category,
        # length of label_start_ids and label_end_ids is not equal
        num_label = max(len(label_start_ids), len(label_end_ids))

        num_correct = torch.tensor(num_correct, device=self.device) 
        num_infer = torch.tensor(num_infer, device=self.device)
        num_label = torch.tensor(num_label, device=self.device)

        return num_correct, num_infer, num_label

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        total_gold, total_pred, right_pred = self.num_label_spans.float(), self.num_infer_spans.float(), self.num_correct_spans.float()
        prec = torch.tensor(0.0, device=self.device) if total_pred == 0 else (right_pred / total_pred)
        rec = torch.tensor(0.0, device=self.device) if total_gold == 0 else (right_pred / total_gold)
        f1 = torch.tensor(0.0, device=self.device) if prec + rec == 0 else (2 * rec * prec) / (prec + rec)

        return prec, rec, f1

