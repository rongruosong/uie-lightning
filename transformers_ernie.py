# coding=utf-8

from typing import Optional, Tuple, Union, Dict
import torch 
from torch import nn, Tensor
from transformers import ErnieModel, ErniePreTrainedModel, ErnieConfig

from loss import uie_loss_func


class UIETorch(ErniePreTrainedModel):
    #  _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    def __init__(self, config):
        super(UIETorch, self).__init__(config)
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        hidden_size = config.hidden_size
        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        input_ids: Optional[Tensor] = None, 
        token_type_ids: Optional[Tensor] = None, 
        pos_ids: Optional[Tensor] = None, 
        att_mask: Optional[Tensor] = None,
        start_ids: Optional[Tensor] = None,
        end_ids: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
    
        if (start_ids is None and end_ids is not None) or (start_ids is not None and end_ids is None):
            raise ValueError("Both of start_ids and end_ids must be not None simultanously!")
        outputs = self.ernie(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=att_mask)

        sequence_output = outputs[0]
        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_probs = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_probs = self.sigmoid(end_logits)
        res_outputs = {
            'start_prob': start_probs, 
            'end_prob': end_probs
        }
        if start_ids is not None:
            loss = uie_loss_func((start_probs, end_probs), (start_ids, end_ids))
            res_outputs['loss'] = loss

        return res_outputs

if __name__ == '__main__':
    config_path = '/gemini/pretrain/model_config.json'
    model_path = '/gemini/pretrain/pytorch_model.bin'
    config = ErnieConfig.from_pretrained(config_path)
    model = UIETorch.from_pretrained(model_path, config=config)
    print(config)