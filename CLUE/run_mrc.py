import argparse
import collections
import json
import os
import random

import numpy as np
import torch
from preprocess.cmrc2018_evaluate import get_eval
from tools import official_tokenization as tokenization, utils
from tqdm import tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizerFast,
)
import sys
from tools.common import seed_everything
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

sys.path.append("../")
from utils.modeling import GAUConfig, GAUForQuestionAnswering
from utils.roformer.configuration_roformer import RoFormerConfig
from utils.roformer.tokenization_roformer import RoFormerTokenizer
from utils.roformer.modeling_roformer import RoFormerForQuestionAnswering

WEIGHTS_NAME = "pytorch_model.bin"
MODEL_CLASSES = {
    # RoFormer, GAU
    'gau': (GAUConfig, GAUForQuestionAnswering, BertTokenizerFast),
    'roformer': (RoFormerConfig, RoFormerForQuestionAnswering, RoFormerTokenizer),
    'roformerv2': (RoFormerConfig, RoFormerForQuestionAnswering, RoFormerTokenizer)
}


def evaluate(model, args, eval_examples, eval_features, device, global_steps, best_f1, best_em, best_f1_em):
    print("***** Eval *****")
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    output_prediction_file = os.path.join(args.checkpoint_dir,
                                          "predictions_steps" + str(global_steps) + ".json")
    output_nbest_file = output_prediction_file.replace('predictions', 'nbest')
    
    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.n_batch, shuffle=False)
    
    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            batch_start_logits, batch_end_logits = output.start_logits, output.end_logits
        
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    
    write_predictions(eval_examples, eval_features, all_results,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)
    
    tmp_result = get_eval(args.dev_file, output_prediction_file)
    tmp_result['STEP'] = global_steps
    with open(args.log_file, 'a') as aw:
        aw.write(json.dumps(tmp_result) + '\n')
    print(tmp_result)
    
    if float(tmp_result['F1']) > best_f1:
        best_f1 = float(tmp_result['F1'])
    
    if float(tmp_result['EM']) > best_em:
        best_em = float(tmp_result['EM'])
    
    if float(tmp_result['F1']) + float(tmp_result['EM']) > best_f1_em:
        best_f1_em = float(tmp_result['F1']) + float(tmp_result['EM'])
        utils.torch_save_model(model, args.checkpoint_dir,
                               {'f1': float(tmp_result['F1']), 'em': float(tmp_result['EM'])}, max_save_num=1)
    
    model.train()
    
    return best_f1, best_em, best_f1_em


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')
    
    # training parameter
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.05)
    parser.add_argument("--weight_decay", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--seed', type=list, default=[1234])
    parser.add_argument('--float16', action='store_true', default=False)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--eval_epochs', type=float, default=0.25)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--max_seq_length', type=int, default=512)
    
    # data dir
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument('--train_dir', type=str,
                        default="/root/autodl-tmp/GAU-PyTorch/CLUE/mrc_data/CMRC2018/train_features.json")
    parser.add_argument('--dev_dir1', type=str,
                        default="/root/autodl-tmp/GAU-PyTorch/CLUE/mrc_data/CMRC2018/dev_examples.json")
    parser.add_argument('--dev_dir2', type=str,
                        default="/root/autodl-tmp/GAU-PyTorch/CLUE/mrc_data/CMRC2018/dev_features.json")
    parser.add_argument('--train_file', type=str,
                        default="/root/autodl-tmp/GAU-PyTorch/CLUE/mrc_data/CMRC2018/train.json")
    parser.add_argument('--dev_file', type=str, default="/root/autodl-tmp/GAU-PyTorch/CLUE/mrc_data/CMRC2018/dev.json")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--task_name', type=str, default="cmrc2018")
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')
    
    # use some global vars for convenience
    args = parser.parse_args()
    
    if args.task_name.lower() == 'drcd':
        from preprocess.DRCD_output import write_predictions
        from preprocess.DRCD_preprocess import json2features
    elif args.task_name.lower() == 'cmrc2018':
        from preprocess.cmrc2018_output import write_predictions
        from preprocess.cmrc2018_preprocess import json2features
    else:
        raise NotImplementedError
    
    args.train_dir = args.train_dir.replace('features.json', 'features_' + str(args.max_seq_length) + '.json')
    args.dev_dir1 = args.dev_dir1.replace('examples.json', 'examples_' + str(args.max_seq_length) + '.json')
    args.dev_dir2 = args.dev_dir2.replace('features.json', 'features_' + str(args.max_seq_length) + '.json')
    args = utils.check_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))
    
    # init model
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=True)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    utils.torch_show_all_params(model)
    
    # load data
    print('loading data...')
    if not os.path.exists(args.train_dir):
        json2features(args.train_file, [args.train_dir.replace('_features_', '_examples_'), args.train_dir],
                      tokenizer, is_training=True,
                      max_seq_length=args.max_seq_length)
    
    if not os.path.exists(args.dev_dir1) or not os.path.exists(args.dev_dir2):
        json2features(args.dev_file, [args.dev_dir1, args.dev_dir2], tokenizer, is_training=False,
                      max_seq_length=args.max_seq_length)
    
    train_features = json.load(open(args.train_dir, 'r'))
    dev_examples = json.load(open(args.dev_dir1, 'r'))
    dev_features = json.load(open(args.dev_dir2, 'r'))
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    
    steps_per_epoch = len(train_features) // args.n_batch
    eval_steps = int(steps_per_epoch * args.eval_epochs)
    dev_steps_per_epoch = len(dev_features) // args.n_batch
    if len(train_features) % args.n_batch != 0:
        steps_per_epoch += 1
    if len(dev_features) % args.n_batch != 0:
        dev_steps_per_epoch += 1
    total_steps = steps_per_epoch * args.train_epochs
    
    print('steps per epoch:', steps_per_epoch)
    print('total steps:', total_steps)
    print('warmup steps:', int(args.warmup_rate * total_steps))
    
    F1s = []
    EMs = []
    # 存一个全局最优的模型
    best_f1_em = 0
    
    for seed_ in args.seed:
        best_f1, best_em = 0, 0
        with open(args.log_file, 'a') as aw:
            aw.write('===================================' +
                     'SEED:' + str(seed_)
                     + '===================================' + '\n')
        print('SEED:', seed_)
        
        # Set seed
        seed_everything(seed_)
        
        if args.float16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_rate * total_steps),
            num_training_steps=total_steps
        )
        
        all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f['input_mask'] for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features], dtype=torch.long)
        
        seq_len = all_input_ids.shape[1]
        
        # true label
        all_start_positions = torch.tensor([f['start_position'] for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f['end_position'] for f in train_features], dtype=torch.long)
        
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        train_dataloader = DataLoader(train_data, batch_size=args.n_batch, shuffle=True)
        
        print('***** Training *****')
        model.train()
        global_steps = 1
        best_em = 0
        best_f1 = 0
        for i in range(int(args.train_epochs)):
            print('Starting epoch %d' % (i + 1))
            total_loss = 0
            iteration = 1
            with tqdm(total=steps_per_epoch, desc='Epoch %d' % (i + 1)) as pbar:
                for step, batch in enumerate(train_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    loss = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                                 start_positions=start_positions, end_positions=end_positions).loss
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (iteration + 1e-5))})
                    pbar.update(1)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                    
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_steps += 1
                    iteration += 1
                    
                    if global_steps % eval_steps == 0:
                        best_f1, best_em, best_f1_em = evaluate(model, args, dev_examples, dev_features, device,
                                                                global_steps, best_f1, best_em, best_f1_em)
        
        F1s.append(best_f1)
        EMs.append(best_em)
        
        # release the memory
        del model
        del optimizer
        torch.cuda.empty_cache()
    
    print('Mean F1:', np.mean(F1s), 'Mean EM:', np.mean(EMs))
    print('Best F1:', np.max(F1s), 'Best EM:', np.max(EMs))
    with open(args.log_file, 'a') as aw:
        aw.write('Mean(Best) F1:{}({})\n'.format(np.mean(F1s), np.max(F1s)))
        aw.write('Mean(Best) EM:{}({})\n'.format(np.mean(EMs), np.max(EMs)))
