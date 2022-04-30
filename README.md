# GAU-PyTorch
## 一、Describtion
&emsp;&emsp;`PyTorch`版本的魔改 [《Transformer Quality in Linear Time》](https://arxiv.org/abs/2202.10447)在中文 NLU 任务上的复现、评测与压缩试验。  
&emsp;&emsp;参考：JunnYu 的 [实现](https://github.com/JunnYu/GAU-alpha-pytorch) 与苏神的 [实现](https://github.com/ZhuiyiTechnology/GAU-alpha) 与 [笔记](https://spaces.ac.cn/archives/8934)

## 二、模型概述
&emsp;&emsp;原论文提出了一种 ***将 Attention 与 FFN 相结合*** 的方案：GAU。  
&emsp;&emsp;标准的 FFN 是两层 MLP ：  
$$
\begin{equation}\boldsymbol{O}=\phi(\boldsymbol{X}\boldsymbol{W}_u)\boldsymbol{W}_o\end{equation}
$$
&emsp;&emsp;其中 $$\boldsymbol{X}\in\mathbb{R}^{n\times d},\boldsymbol{W}_u\in\mathbb{R}^{d\times d_{ff}},\boldsymbol{W}_o\in\mathbb{R}^{d_{ff}\times d}$$。而后，有研究表明将 FFN 换成如下的 GLU 效果更好：  
$$
\begin{equation}\boldsymbol{O}=(\boldsymbol{U}\odot\boldsymbol{V})\boldsymbol{W}_o,\quad \boldsymbol{U}=\phi_u(\boldsymbol{X}\boldsymbol{W}_u),\quad\boldsymbol{V}=\phi_v(\boldsymbol{X}\boldsymbol{W}_v)\end{equation}
$$  
&emsp;&emsp;其中 $$\boldsymbol{W}_u,\boldsymbol{W}_v\in\mathbb{R}^{d\times e},\odot $$ 为按位相乘。  
&emsp;&emsp;我们知道，在 Transformer 结构中，Attention 的作用是捕捉 Token-to-Token 间的关系；而 FFN 则是增强模型的非线性表达能力。而 GLU 只是用 *自身* 去 gate *自身*，忽略了 Token 间关系，那么一个自然的想法是用 Token-to-Token 的关系也即 ***注意力矩阵 A*** 去 gate：  
$$
\begin{equation}\boldsymbol{O}=(\boldsymbol{U}\odot\boldsymbol{A}\boldsymbol{V})\boldsymbol{W}_o\label{eq:mix}\end{equation}
$$  
&emsp;&emsp;在原论文中，注意力矩阵被简化至了如下的形式：  
$$
\begin{equation}\boldsymbol{A}=\text{relu}^2\left(\frac{\mathcal{Q}(\boldsymbol{Z})\mathcal{K}(\boldsymbol{Z})^{\top}}{n}\right)=\frac{1}{n^2}\text{relu}^2\left(\mathcal{Q}(\boldsymbol{Z})\mathcal{K}(\boldsymbol{Z})^{\top}\right),\quad \boldsymbol{Z}=\phi_z(\boldsymbol{X}\boldsymbol{W}_z)\label{eq:relu-att}\end{equation}
$$  
&emsp;&emsp;其中 $$\boldsymbol{W}_z\in\mathbb{R}^{d\times s}$$，$$s$$ 即注意力的 head_size，文中取了 $$s=128$$ ，而 $$\mathcal{Q},\mathcal{K}$$ 是简单的仿射变换（像 Layer Norm 中的乘 $$\gamma$$ 加  $$\beta$$），$$relu^2$$ 则是 $$relu$$ 后再平方。  
&emsp;&emsp;而后每一层只用一个 GAU 单元就行了，原来一层的计算量大约等于两层 GAU。而且 GAU 只需要一个头就行了。  
&emsp;&emsp;原论文的主要思路便是这些了，很简洁但是也有很多不合理的地方：比如说缩放因子取的是 $$n^2$$ ？？在长序列的时候是不是太小了？还有就是 $$relu^2$$​ ，虽然说这是 NAS 搜出来的，但是相比 softmax 感觉不是太好。这部分 [苏神也分析到了](https://spaces.ac.cn/archives/9019)。也可以从稀疏性去考虑：原始 softmax 不仅仅带来的是非负性与因子的缩放，更代表着将 Token 间的关系集中在少数几个 Token 上，这种注意力的“集中”、更多信息的引入在数学上表现为秩的增加；而 $$relu^2$$​ 在负方向直接归零了，而且 RoPE 具有一定的远程衰减功能，在相对距离足够大时，即便是初始阶段经过 RoPE 之后的内积结果平均来说至少有一半是负的，所以这种 Attention 最后有相当一部分是零，导致秩降低的可能性更大，包含的信息很可能是减少的。  
鉴于此，本文使用了苏神提出的改进版的 softmax：  
$$
\begin{equation}\boldsymbol{A}=softmax\left(\frac{\log_{512} n}{\sqrt{d}}\mathcal{Q}(\boldsymbol{Z})\mathcal{K}(\boldsymbol{Z})^{\top}\right)\end{equation}
$$

## 三、TODO
- 创建显存、训练、推理速度比较代码  

## 四、更新
- 2022/04/27 添加模型笔记、显存、推理速度比较结果。   
- 2022/04/23 增加 CLUE 评测  
- 2022/04/18 两种归一化策略比较  
- 2022/04/14 重构预训练代码  

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
&emsp;&emsp;与 FLASH 原论文提出的：  
$$
squared\_relu=\frac{1}{n} relu^2\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
$$
&emsp;&emsp;在预训练阶段：squared relu 出现了较为严重的波动，几次 checkpoint 也未能恢复正常。排查原因后发现是波动附近 batch 的 sql_len 与其他 batch 接近 512 的长度有较大差异。这进一步说明了 squared relu 在样本长度方面的迁移能力不好。而苏神的 softmax plus 则训练稳定，效果较好，原因与推导详见 [blog](https://spaces.ac.cn/archives/9019)  

## 七、测试
&emsp;&emsp;以下均是在 A40*1 取得的结果。  
### 7.1 CLUE 分类测试
&emsp;&emsp;在 CLUE 分类数据集上对 [RoFormerV1](https://huggingface.co/junnyu/roformer_chinese_base)、RoFormerV2（多任务）、RoFormerV2（MLM，3W步）、GAU（3W步）、GAU（完整训练）（以上均为 Base 模型）进行对比：  
|  模型  | AFQMC  | CMNLI | CSL | IFLYTEK | TNews | WSC | Score |
|  :--:  | :--:  | :--:  | :--:  | :--:  | :--:  | :--:  | :--:  |
| RoFormerV1 | 74.21 | 81.50 | 83.13 | 60.17 | 58.07 | 83.22 | 73.38 |
| RoFormerV2 | **76.16** | 81.41 | **85.97** | **63.64** | **59.39** | **85.53** | **75.35** |
| GAU(3W) | 73.14 | 76.73 | 80.44 | 59.81 | 54.66 | 75.9 | 70.19 |
| RoFormerV2(3W) | 73.97 | 77.82 | 79.65 | 58.73 | 53.3 | 76.41 | 69.96 |
| GAU(Full) | 74.51 | **81.97** | 83.7 | 62.72 | 57.93 | 82.89 | 73.95 |

&emsp;&emsp;可见 GAU 虽比拟不了 RoFormerV2 的结果，但超出 V1 约 0.6 个点，而且比对两个训练 3W 步的模型可知，GAU 的拟合能力是不逊于 RoFormerV2 的。如果再仔细设计一下预训练，增多语料很有可能会取得更好的结果。  

### 7.2 显存、速度对比
&emsp;&emsp;以下是在 ***CLUE CMNLI*** 任务上取 bs=8 sql_len=512 的推理速度：  
|  模型   | Time Cost  | Percent |
|  :--:  | :--:  | :--:  |
| RoFormerV1 | 03:37:48 | 1.64× |
| RoFormerV2 | 03:03:09 | 1.379× |
| GAU | 02:12:48 | 1× |

&emsp;&emsp;以下是在 ***CLUE CMNLI*** 任务上取 bs=64 sql_len=512 的最大显存占用：  
|  模型   | VRam Cost  | Percent |
|  :--:  | :--:  | :--:  |
| RoFormerV1 | 18487M | 100% |
| RoFormerV2 | 15807M | 85.5% |
| GAU | 17453M | 94.4% |

&emsp;&emsp;以下是各模型的参数量对比：（XForSequenceClassification）  
|  模型   | Parametrers  | Percent |
|  :--:  | :--:  | :--:  |
| RoFormerV1 | 124,147,970 | 128.63% |
| RoFormerV2 | 94,777,090 | 98.20% |
| GAU | 96,519,170 | 100% |

