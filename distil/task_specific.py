import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizerFast,
)
import sys

sys.path.append("../")
sys.path.append("../CLUE")
sys.path.append("/root/autodl-tmp/TextBrewer/src")
from metrics.clue_compute_metrics import compute_metrics
from processors import clue_processors as processors
from processors import clue_output_modes as output_modes
from processors import clue_convert_examples_to_features as convert_examples_to_features
from tools.common import seed_everything
from tools.common import init_logger, logger
from tools.progressbar import ProgressBar

from utils.modeling import GAUConfig, GAUForMaskedLM, GAUForSequenceClassification
from utils.roformer.configuration_roformer import RoFormerConfig
from utils.roformer.tokenization_roformer import RoFormerTokenizer
from utils.roformer.modeling_roformer import RoFormerForSequenceClassification

import textbrewer
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig

WEIGHTS_NAME = "pytorch_model.bin"
TEACHER_MODEL_CLASSES = {
    # RoFormer, GAU
    'roformerv2': (RoFormerConfig, RoFormerForSequenceClassification, RoFormerTokenizer)
}


def TeacherAdaptor(batch, model_outputs, with_logits=False, with_mask=False):
    dict_obj = {'hidden': model_outputs[2]}
    if with_mask:
        dict_obj['inputs_mask'] = batch[1]
    if with_logits:
        dict_obj['logits'] = (model_outputs[1],)
    return dict_obj


def StudentAdaptor(batch, model_outputs, with_logits=False, with_mask=False):
    dict_obj = {'hidden': model_outputs[2]}
    if with_mask:
        dict_obj['inputs_mask'] = batch[1]
    if with_logits:
        dict_obj['logits'] = (model_outputs[1],)
    return dict_obj


def divide_parameters(args, named_parameters, lr=None):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    decay_parameters_names = list(zip(*[(p, n) for n, p in named_parameters if not any((di in n) for di in no_decay)]))
    no_decay_parameters_names = list(zip(*[(p, n) for n, p in named_parameters if any((di in n) for di in no_decay)]))
    param_group = []
    if len(decay_parameters_names) > 0:
        decay_parameters, decay_names = decay_parameters_names
        # print ("decay:",decay_names)
        if lr is not None:
            decay_group = {'params': decay_parameters, 'weight_decay': args.weight_decay, 'lr': lr}
        else:
            decay_group = {'params': decay_parameters, 'weight_decay': args.weight_decay}
        param_group.append(decay_group)
    
    if len(no_decay_parameters_names) > 0:
        no_decay_parameters, no_decay_names = no_decay_parameters_names
        # print ("no decay:", no_decay_names)
        if lr is not None:
            no_decay_group = {'params': no_decay_parameters, 'weight_decay': 0.0, 'lr': lr}
        else:
            no_decay_group = {'params': no_decay_parameters, 'weight_decay': 0.0}
        param_group.append(no_decay_group)
    
    assert len(param_group) > 0
    return param_group


def get_args_parser():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(TEACHER_MODEL_CLASSES.keys()))
    parser.add_argument("--teacher_model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--student_model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir", default=None, type=str, required=True)
    
    ## Other parameters
    parser.add_argument("--student_gau_layers", default=6, type=int)
    parser.add_argument("--temperature", default=10, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument('--logging_steps', type=int, default=200,
                        help="Log every X updates steps.")
    parser.add_argument('--ckpt_frequency', type=int, default=1,
                        help="stores model weights ckpt_frequency times every epoch.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser


def load_data(args, task, tokenizer, phase='train'):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_student_{}_{}_{}'.format(
        phase,
        str(args.max_seq_length),
        str(task)))
    
    if os.path.exists(cached_features_file):
        logger.info("Loading student features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating student features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and 'roberta' in args.model_type:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        
        if phase == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif phase == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        logger.info("Saving student features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset


def collate_fn(batch):
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'token_type_ids': all_token_type_ids,
        'labels': all_labels
    }


def evaluate(args, model, eval_dataset, step):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)
    # Eval!
    logger.info("********* Running evaluation ********")
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch)[1]
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch['labels'].detach().cpu().numpy(), axis=0)
        pbar(step)
    print(' ')
    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(args.task_name, preds, out_label_ids)
    results.update(result)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("******** Eval results ********")
    for key in sorted(result.keys()):
        logger.info(" dev: %s = %s", key, str(result[key]))
    model.train()
    return results


def main(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + 'teacher:{}'.format(args.teacher_model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir + '/teacher:{}-{}.log'.format(args.teacher_model_type, args.task_name))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    
    # Setup device
    if "cuda" in args.device and torch.cuda.is_available():
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")
    
    # Set seed
    seed_everything(args.seed)
    # Prepare CLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors: raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    
    # Prepate teacher model
    args.teacher_model_type = args.teacher_model_type.lower()
    teacher_config_class, teacher_model_class, teacher_tokenizer_class = TEACHER_MODEL_CLASSES[args.teacher_model_type]
    teacher_config = teacher_config_class.from_pretrained(args.teacher_model_name_or_path, num_labels=num_labels,
                                                          finetuning_task=args.task_name, output_attentions=True,
                                                          output_hidden_states=True)
    teacher_tokenizer = teacher_tokenizer_class.from_pretrained(args.teacher_model_name_or_path,
                                                                do_lower_case=args.do_lower_case)
    teacher_model = teacher_model_class.from_pretrained(args.teacher_model_name_or_path, config=teacher_config)
    teacher_model.to(args.device)
    logger.info("Teacher model prepared.")
    
    # Prepare student model
    student_config = GAUConfig.from_pretrained(args.student_model_name_or_path, num_labels=num_labels,
                                               finetuning_task=args.task_name,
                                               num_hidden_layers=args.student_gau_layers,
                                               output_attentions=True,
                                               output_hidden_states=True)
    student_tokenizer = BertTokenizerFast.from_pretrained(args.student_model_name_or_path,
                                                          do_lower_case=args.do_lower_case)
    
    # Initialize the student GAU model by using pretrained weights.
    # student layer: n <----- teacher layer: (4n+3)
    student_model = GAUForSequenceClassification(config=student_config)
    state_dict = torch.load(args.student_model_name_or_path + WEIGHTS_NAME)
    new_dict = {}
    for k, v in state_dict.items():
        if "gau.encoder.layer." not in k:
            new_dict[k] = v
        else:
            keys = k.split(".")
            if int(keys[3]) % 4 == 3 and int(k.split(".")[3]) != 0:
                keys[3] = str((int(keys[3]) - 3) // 4)
                new_dict[".".join(keys)] = v
    student_model.load_state_dict(new_dict, strict=False)
    student_model.to(args.device)
    logger.info("Student model prepared.")
    
    # Compare the parameters num
    logger.info("\nTeacher_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(teacher_model, max_level=3)
    logger.info(result)
    
    logger.info("Sudent_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(student_model, max_level=3)
    logger.info(result)
    
    # Prepare data
    train_dataset = load_data(args, args.task_name, student_tokenizer, phase='train')
    eval_dataset = load_data(args, args.task_name, student_tokenizer, phase='dev')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)
    num_train_steps = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)
    
    # Prepate optimizer, lr scheduler
    params = list(student_model.named_parameters())
    all_trainable_params = divide_parameters(args, params, lr=args.learning_rate)
    optimizer = AdamW(all_trainable_params, lr=args.learning_rate)
    scheduler_class = get_linear_schedule_with_warmup
    scheduler_args = {'num_warmup_steps': int(args.warmup_proportion * num_train_steps),
                      'num_training_steps': num_train_steps}
    
    # Prepare configurations
    train_config = TrainingConfig(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ckpt_frequency=args.ckpt_frequency,
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        fp16=args.fp16,
        device=args.device)
    intermediate_matches = [
        {'layer_T': 0, 'layer_S': 0, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},  # embedding layer
    ]
    distill_config = DistillationConfig(
        temperature=args.temperature,
        intermediate_matches=intermediate_matches)
    logger.info(f"{train_config}")
    logger.info(f"{distill_config}")
    adaptor_T = TeacherAdaptor
    adaptor_S = StudentAdaptor
    distiller = GeneralDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher_model, model_S=student_model,
        adaptor_T=adaptor_T, adaptor_S=adaptor_S)
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Train batch size = %d", args.train_batch_size)
    logger.info("  Num backward steps = %d", num_train_steps)
    
    callback_func = partial(evaluate, eval_dataset=eval_dataset, args=args)
    with distiller:
        distiller.train(optimizer, scheduler_class=scheduler_class, scheduler_args=scheduler_args,
                        dataloader=train_dataloader,
                        num_epochs=args.num_train_epochs, callback=callback_func, max_grad_norm=args.max_grad_norm)


if __name__ == "__main__":
    # for debugging
    """
    args = get_args_parser().parse_args(['--teacher_model_type', 'roformerv2',
                                         '--data_dir', '/root/autodl-tmp/GAU-PyTorch/CLUE/CLUEdatasets/tnews',
                                         '--teacher_model_name_or_path',
                                         '/root/autodl-tmp/models/roformerv2_chinese_base',
                                         '--student_model_name_or_path', '/root/autodl-tmp/models/GAU-Base-Full/',
                                         '--task_name', 'tnews',
                                         '--output_dir', '/root/autodl-tmp/GAU-PyTorch/distil/output/tnews/',
                                         '--log_dir', '/root/tf-logs/tnews/128',
                                         '--max_seq_length', '512',
                                         '--train_batch_size', '128',
                                         '--learning_rate', '3e-5',
                                         '--do_lower_case',
                                         '--overwrite_output_dir',
                                         '--logging_steps', '500',
                                         '--warmup_proportion', '0.1',
                                         '--num_train_epochs', '3'
                                         ])
    """
    args = get_args_parser().parse_args()
    main(args)
