# coding=utf-8
import torch
from torch.optim import AdamW
import math
from lightning.fabric import Fabric
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup, ErnieConfig
from transformers_ernie import UIETorch
from dataset import create_data_loader
from train import Trainer
from lightning.fabric.loggers import TensorBoardLogger
from train_argparse import parse_args
import warnings
warnings.filterwarnings("ignore")

def main():
    args = parse_args()

    # fabric
    logger = TensorBoardLogger(args.log_dir)
    fabric = Fabric(loggers=logger)
    fabric.seed_everything(args.seed)
    fabric.print('world_size {}, global_rank {}'.format(fabric.world_size, fabric.global_rank))
    torch.set_float32_matmul_precision('high')
    # model
    fabric.print('load model ...')
    tokenizer = BertTokenizerFast.from_pretrained(args.vocab_path)
    config = ErnieConfig.from_pretrained(args.config_path)
    model = UIETorch.from_pretrained(args.pretrained_model_path, config=config)
    fabric.print('model loading finish!')

    # dataloader
    fabric.print('load train data ...')
    train_loader = create_data_loader(args.train_data_path, 
        tokenizer=tokenizer, 
        batch_size=args.train_batch_size, 
        max_seq_len=args.seq_length,
        world_size=fabric.world_size
    )
    fabric.print('load val data ...')
    val_loader = create_data_loader(args.test_data_path, 
        tokenizer=tokenizer, 
        batch_size=args.test_batch_size, 
        max_seq_len=args.seq_length,
        world_size=1
    )

    # optimizer and scheduler
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    estimated_stepping_batches = math.ceil(len(train_loader) / args.grad_accum_steps) * max(args.max_epochs, 1)
    num_warmup_steps = args.warmup * estimated_stepping_batches
    """
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=estimated_stepping_batches,
    )
    """
    fabric.print('total steps: {}'.format(estimated_stepping_batches))

    # train
    fabric.print('begin to fit model')
    trainer = Trainer(args, fabric, optimizer=optimizer)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
    