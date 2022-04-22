# GAU-PyTorch
## 一、Describtion
&emsp;&emsp;`pytorch`版本的魔改 [《Transformer Quality in Linear Time》](https://arxiv.org/abs/2202.10447)  
&emsp;&emsp;参考：JunnYu 的 [实现](https://github.com/JunnYu/GAU-alpha-pytorch) 与苏神的 [实现](https://github.com/ZhuiyiTechnology/GAU-alpha)

## 二、存在的问题

## 三、TODO
- 创建 CLUE 测试代码  

## 四、更新
- 2022/04/22 重构预训练代码  

## 五、Pretrain
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
TRAIN_DIR=/root/autodl-tmp/FLASHQuad_pytorch/clue_small_wwm_data
OUTPUT_DIR=/root/autodl-tmp/FLASHQuad_pytorch/wwm_flash_small/
BATCH_SIZE=256
ACCUMULATION=4
LR=4e-4
python run_mlm_wwm.py \
    --do_train \
    --tokenizer_name junnyu/roformer_chinese_char_base \
    --train_dir $TRAIN_DIR \
    --output_dir $OUTPUT_DIR \
    --logging_dir /root/tf-logs \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATION \
    --learning_rate $LR \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_steps 250000 \
    --warmup_steps 5000 \
    --logging_steps 100 \
    --save_steps 5000 \
    --seed 1234 \
    --max_grad_norm 3.0 \
    --dataloader_num_workers 6 \
    --fp16 \
    --overwrite_output_dir
```

## 六、测试
### 6.1 MLM测试
```python
# flashquad
import torch
from flash import FLASHQuadForMaskedLM
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("junnyu/flashquad_small_wwm_cluecorpussmall")
model = FLASHQuadForMaskedLM.from_pretrained("junnyu/flashquad_small_wwm_cluecorpussmall")
model.eval()
text = "天气预报说今天的天[MASK]很好，那么我[MASK]一起去公园玩吧！"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    pt_outputs = model(**inputs).logits[0]

pt_outputs_sentence = "pytorch: "
for i, id in enumerate(tokenizer.encode(text)):
    if id == tokenizer.mask_token_id:
        val,idx = pt_outputs[i].softmax(-1).topk(k=5)
        tokens = tokenizer.convert_ids_to_tokens(idx)
        new_tokens = []
        for v,t in zip(val.cpu(),tokens):
            new_tokens.append(f"{t}+{round(v.item(),4)}")
        pt_outputs_sentence += "[" + "||".join(new_tokens) + "]"
    else:
        pt_outputs_sentence += "".join(
            tokenizer.convert_ids_to_tokens([id], skip_special_tokens=True))
print(pt_outputs_sentence)
# pytorch: 天气预报说今天的天[气+0.9948||空+0.0011||色+0.0007||候+0.0004||势+0.0003]很好，那么我[就+0.4915||们+0.4186||也+0.0753||还+0.0021||都+0.0016]一起去公园玩吧！

# flash
import torch
from flash import FLASHForMaskedLM
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("junnyu/flash_small_wwm_cluecorpussmall")
model = FLASHForMaskedLM.from_pretrained("junnyu/flash_small_wwm_cluecorpussmall")
model.eval()
text = "天气预报说今天的天[MASK]很好，那么我[MASK]一起去公园玩吧！"
inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=512,  return_token_type_ids=False) #这里必须是512，不然结果可能不对。
with torch.no_grad():
    pt_outputs = model(**inputs).logits[0]

pt_outputs_sentence = "pytorch: "
for i, id in enumerate(tokenizer.encode(text)):
    if id == tokenizer.mask_token_id:
        val,idx = pt_outputs[i].softmax(-1).topk(k=5)
        tokens = tokenizer.convert_ids_to_tokens(idx)
        new_tokens = []
        for v,t in zip(val.cpu(),tokens):
            new_tokens.append(f"{t}+{round(v.item(),4)}")
        pt_outputs_sentence += "[" + "||".join(new_tokens) + "]"
    else:
        pt_outputs_sentence += "".join(
            tokenizer.convert_ids_to_tokens([id], skip_special_tokens=True))
print(pt_outputs_sentence)
# pytorch: 天气预报说今天的天[气+0.9938||天+0.0017||空+0.0011||晴+0.0007||阳+0.0002]很好，那么我[们+0.9367||就+0.0554||也+0.0041||俩+0.0005||还+0.0004]一起去公园玩吧！
```

### 6.2 CLUE 测试