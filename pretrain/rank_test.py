import os
import random
import time
import datetime
import numpy as np
import torch
from typing import Optional
from datasets import Dataset
from dataclasses import dataclass, field
from transformers import (
    BertTokenizerFast,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AdamW, get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import sys

sys.path.append("../")
from utils.modeling import GAUConfig, GAUForMaskedLM, GAUForSequenceClassification
from utils.roformer.configuration_roformer import RoFormerConfig
from utils.roformer.tokenization_roformer import RoFormerTokenizer
from utils.roformer.modeling_roformer import RoFormerForSequenceClassification

class Mydataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = RoFormerTokenizer.from_pretrained("/root/autodl-tmp/models/roformerv2_chinese_base")
    
    def __len__(self):
        return len(data)
    
    def __getitem__(self, index):
        text = self.tokenizer(
            self.data[index],
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return text


def get_data():
    text = ""
    data = []
    with open("/root/autodl-tmp/GAU-PyTorch/text.txt", "r", encoding="utf-8") as f:
        string = f.read().replace("\n", "")
        text += string
    
    for i in range(64):
        data.append(text[5000 * i: 5000 * i + 5000])
    return data


if __name__ == "__main__":
    config = RoFormerConfig.from_pretrained("/root/autodl-tmp/models/roformerv2_chinese_base", max_position_embeddings=2048, num_labels=2)
    model = RoFormerForSequenceClassification(config).cuda()
    model.load_state_dict(torch.load("/root/autodl-tmp/models/roformerv2_chinese_base/pytorch_model.bin"), strict=False)
    data = get_data()
    dataset = Mydataset(data)
    dataloader = DataLoader(dataset, batch_size=8)
    labels = torch.LongTensor([
        1, 1, 1, 1
    ]).cuda()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=10*8
    )
    start = time.time()
    i = 0
    for epoch in range(10):
        for batch in dataloader:
            i += 1
            batch = {k: torch.squeeze(v.cuda(), dim=1) for k, v in batch.items()}
            output = model(**batch, labels=labels)
            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            print(i)
    delta = time.time() - start
    print(delta)
