import os
import sys
import time
import logging

import torch
import transformers

from typing import Optional
from datasets import Dataset
from dataclasses import dataclass, field
from transformers import (
    BertTokenizerFast,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from mlm_trainer import Trainer
import sys

sys.path.append("../")
from utils.modeling import GAUConfig, GAUForMaskedLM, GAUForSequenceClassification
from utils.roformer.configuration_roformer import RoFormerConfig
from utils.roformer.tokenization_roformer import RoFormerTokenizer
from utils.roformer.modeling_roformer import RoFormerForMaskedLM

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    # RoFormer, GAU
    'gau': (GAUConfig, GAUForMaskedLM, BertTokenizerFast),
    'roformer': (RoFormerConfig, RoFormerForMaskedLM, RoFormerTokenizer),
    'roformerv2': (RoFormerConfig, RoFormerForMaskedLM, RoFormerTokenizer)
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    
    tokenizer_name: Optional[str] = field(
        default="junnyu/roformer_chinese_char_base",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    activation_function: Optional[str] = field(
        default="softmax_plus",
        metadata={
            "help": "attention normalize functions"
        },
    )
    scaling_factor: Optional[str] = field(
        default="n",
        metadata={
            "help": "scaling_factor"
        },
    )
    model_type: Optional[str] = field(
        default="gau",
        metadata={
            "help": "model_type"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    train_dir: Optional[str] = field(
        default="/root/autodl-tmp/GAU-PyTorch/clue_small_wwm_data",
        metadata={"help": "The input training data file."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated. Default to the max input length of the model."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=12,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={
            "help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if (
            os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(
            training_args.local_rank) else logging.WARN
    )
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # download the dataset.
    # ??????clue_wwm_13g?????????
    datasets = Dataset.load_from_disk(data_args.train_dir)
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    config = config_class.from_pretrained(model_args.tokenizer_name)
    logger.info(model_args.model_type + " Model config %s", config)
    # tokenizer?????????roformer_chinese_char_base
    tokenizer = tokenizer_class.from_pretrained(model_args.tokenizer_name)
    model = model_class(config)
    # model.load_state_dict(torch.load("/root/autodl-tmp/models/GAU/pytorch_model.bin"))
    model.resize_token_embeddings(len(tokenizer))
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = datasets.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    padding = "max_length" if data_args.pad_to_max_length else False
    
    def tokenize_function(examples):
        # Remove empty lines
        texts = []
        chinese_ref = []
        for text, ref in zip(examples["text"], examples["chinese_ref"]):
            if len(text) > 0 and not text.isspace():
                texts.append(text.strip())
                chinese_ref.append(ref)
        examples["text"] = texts
        examples["chinese_ref"] = chinese_ref
        data = tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        data["text"] = texts
        data["chinese_ref"] = chinese_ref
        return data
    
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not data_args.overwrite_cache,
        new_fingerprint="clue_13g_small_roformer_wwm",
    )
    
    training_args.remove_unused_columns = False
    
    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # trainer.add_callback(LoggingCallback(save_interval=training_args.save_interval))
    # Training
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None
    
    logger.info("Training a model...")
    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    train_time = time.time() - start_time
    logger.info(f"Training time: {train_time}")
    trainer.save_model()  # Saves the tokenizer too for easy upload
    
    output_train_file = os.path.join(
        training_args.output_dir, "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
        
        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )


if __name__ == "__main__":
    main()
