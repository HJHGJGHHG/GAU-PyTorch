# GAU-PyTorch
## 一、Describtion
&emsp;&emsp;`pytorch`版本的魔改 [《Transformer Quality in Linear Time》](https://arxiv.org/abs/2202.10447)  
&emsp;&emsp;参考：JunnYu 的 [实现](https://github.com/JunnYu/GAU-alpha-pytorch) 与苏神的 [实现](https://github.com/ZhuiyiTechnology/GAU-alpha)

## 二、模型概述

## 三、TODO
- 创建显存、训练、推理速度比较代码  

## 四、更新
- 2022/04/24 增加 CLUE 评测  
- 2022/04/23 两种归一化策略比较  
- 2022/04/22 重构预训练代码  

## 五、Pretrain
&emsp;&emsp;WWM，结巴分词  
### 5.1 准备数据
&emsp;&emsp;基于 ***CLUECorpusSmall***，数据处理教程 [来源](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/data_tools/README.md)
&emsp;&emsp;**数据集简介**：可用于语言建模、预训练或生成型任务等，数据量超过 14G，近 4000 个定义良好的 txt 文件、50 亿字。主要部分来自于 nlp_chinese_corpus 项目  
&emsp;&emsp;包含如下子语料库（总共 14G 语料）：新闻语料[news2016zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/6bac09db4e6d4857b6d680d34447457490cb2dbdd8b8462ea1780a407f38e12b?responseContentDisposition=attachment%3B%20filename%3Dnews2016zh_corpus.zip)， 社区互动语料[webText2019zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/83da03f7b4974871a52348b41c16c7e3b34a26d5ca644f558df8435be4de51c3?responseContentDisposition=attachment%3B%20filename%3DwebText2019zh_corpus.zip)，维基百科语料[wiki2019zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/d7a166408d8b4ffdaf4de9cfca09f6ee1e2340260f26440a92f78134d068b28f?responseContentDisposition=attachment%3B%20filename%3Dwiki2019zh_corpus.zip)，评论数据语料[comment2019zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/b66ddd445735408383c42322850ac4bb82faf9cc611447c2affb925443de7a6d?responseContentDisposition=attachment%3B%20filename%3Dcomment2019zh_corpus.zip)。  

(1) **数据集解压**
``` shell
unzip comment2019zh_corpus.zip -d  /root/autodl-tmp/FLASHQuad_pytorch/clue_small_wwm_data/comment2019zh_corpus
unzip news2016zh_corpus.zip    -d  /root/autodl-tmp/FLASHQuad_pytorch/clue_small_wwm_data/news2016zh_corpus  
unzip webText2019zh_corpus.zip -d  /root/autodl-tmp/FLASHQuad_pytorch/clue_small_wwm_data/webText2019zh_corpus
unzip wiki2019zh_corpus.zip    -d  /root/autodl-tmp/FLASHQuad_pytorch/clue_small_wwm_data/wiki2019zh_corpus  
```
(2) **将txt文件转换为jsonl格式**
```shell
cd data
python trans_to_json.py  --input_path /root/autodl-tmp/FLASHQuad_pytorch/clue_small_wwm_data --output_path /root/autodl-tmp/FLASHQuad_pytorch/clue_small_wwm_data/clue_corpus_small_14g.jsonl
```
(3) **使用 rjieba 进行中文分词**
&emsp;&emsp;会得到`refids.txt`和`reftext.txt`两个文件,并组合`refids.txt`和`reftext.txt`两个文件保存成`huggingface`的`dataset`，存放在 clue_small_wwm_data 文件夹下。  
```shell
python run_chinese_ref.py  --model_name junnyu/roformer_chinese_char_base --input_path clue_corpus_small_14g.jsonl
```

### 5.2 开始训练（L-24-H-768）
```bash
TRAIN_DIR=/root/autodl-tmp/GAU-PyTorch/clue_small_wwm_data
OUTPUT_DIR=/root/autodl-tmp/GAU-PyTorch/outputs
BATCH_SIZE=64
ACCUMULATION=4
LR=2e-4
python run_mlm_wwm.py \
  --do_train \
  --tokenizer_name junnyu/roformer_chinese_char_base \
  --train_dir $TRAIN_DIR \
  --output_dir $OUTPUT_DIR \
  --logging_dir /root/tf-logs/$BATCH_SIZE \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $ACCUMULATION \
  --learning_rate $LR \
  --weight_decay 0.01 \
  --adam_epsilon 1e-6 \
  --max_steps 30000 \
  --warmup_steps 3000 \
  --logging_steps 50 \
  --save_steps 3000 \
  --seed 1234 \
  --max_grad_norm 1.0 \
  --dataloader_num_workers 6 \
  --fp16
```

## 六、比较
### 6.1 归一化策略
&emsp;&emsp;根据 [苏神的分析](https://spaces.ac.cn/archives/9019)，采用了两种注意力矩阵的归一化策略：  
$$
softmax\_plus=softmax\left(\frac{\log_{512} n}{\sqrt{d}}QK^{\top}\right)V
$$
&emsp;&emsp;与FLASH 原论文提出的：  
$$
squared\_relu=\frac{1}{n} relu^2\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
$$
&emsp;&emsp;在预训练阶段：squared relu 出现了较为严重的波动，几次 checkpoint 也未能恢复正常。排查原因后发现是波动附近 batch 的 sql_len 与其他 batch 接近 512 的长度有较大差异。这进一步说明了 squared relu 在样本长度方面的迁移能力不好。而苏神的 softmax plus 则训练稳定，效果较好，原因与推导详见 [blog](https://spaces.ac.cn/archives/9019)  

## 七、测试
### 7.1 CLUE 分类测试
&emsp;&emsp;在 CLUE 分类数据集上对 [RoFormerV1](https://huggingface.co/junnyu/roformer_chinese_base)、RoFormerV2（多任务）、RoFormerV2（MLM，3W步）、GAU（3W步）、GAU（完整训练）（以上均为 Base 模型）进行对比：  
|  模型  | AFQMC  | CMNLI | CSL | IFLYTEK | TNews | WSC | Score |
|  :--:  | :--:  | :--:  | :--:  | :--:  | :--:  | :--:  | :--:  |
| RoFormerV1 | 74.21 | 81.50 | 83.13 | 60.17 | 58.07 | 83.22 | 73.38 |
| RoFormerV2 | **76.16** | 81.41 | **85.97** | **63.64** | **59.39** | **85.53** | 75.35 |
| GAU(3W) | 73.14 | 76.73 | 80.44 | 59.81 | 54.66 | 75.9 | 70.19 |
| RoFormerV2(3W) | 73.97 | 77.82 | 79.65 | 58.73 | 53.3 | 76.41 | 69.96 |
| GAU(Full) | 74.51 | **81.97** | 83.7 | 62.72 | 57.93 | 82.89 | 73.95 |

### 7.2 显存、速度对比