# 基于UIE的通用信息抽取(pytorch)
基于paddleNLP的information_extraction/text复现的pytorch版本纯文本信息抽取，以模型UIE为训练底座，进行Finetune。数据加载基于torchdata实现，训练使用lightning Fabric，metirc基于torchmetrics实现，可以便利地支持分布式训练。数据标注请参考:
https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/information_extraction/label_studio_text.md 。

## 代码结构
```
uie-lightning/
├── data/ # 数据样例，根据paddlenlp提供的`军事关系抽取数据集`生成
├── convert_pd_2_torch.py # 将对应的paddle UIE模型转化为pytorch的版本
├── dataconvert.py # 数据集转换，将label_studio标注的原始数据转化为模型训练使用的数据
├── dataloader.py # 继承DataLoader，处理Iterable类型的dataset无法计算loader真实长度的问题
├── dataset.py # torchdata数据预处理
├── finetune.py # 模型微调脚本
├── loss.py # 损失计算函数
├── metrics.py # span评价函数
├── run_finetuning.sh # 运行脚本
├── train_argparse.py # 参数设置
├── train.py # 实现的Trainer类，在Finetune时使用
├── transformers_ernie.py # 基于Transformers实现UIE模型
├── utils.py # 工具函数脚本
└── README.md # 使用说明
```
可以直接使用`bash run_finetuning.sh`运行，在`军事关系抽取数据集`的结果：
|  模型 | Precision | Recall | F1 |
|  :---: | :--------: | :--------: | :--------: |
| `uie-base` |  |  |  |

## Requirements
- pytorch>=1.12.1
- lightning>=2.0
- torchmetrics
- torchdata=0.4
- transformers
- paddle (if you need convert paddle model to torch)

## Todo
- [] upload model parameters to huggingface
- [] adapt for torchdata >= 0.6.0
- [] pytorch2.0
- [] inference serve