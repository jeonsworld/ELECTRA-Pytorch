# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from apex.parallel import DistributedDataParallel as DDP

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm
from utils.optimization import AdamW, WarmupLinearSchedule
from utils.tokenization import BertTokenizer
from utils.pretrain_utils import LMDataset
from models.electra_model import Config, Electra
from apex.optimizers import FusedLAMB
from utils.schedulers import PolyWarmUpScheduler

logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_dir = os.path.join("tensorboard", args.model_name)
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = FusedLAMB(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer=optimizer,
                                     warmup_steps=args.warmup_steps,
                                     t_total=args.num_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level,
                                          cast_model_outputs=torch.float16)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=torch.distributed.get_world_size())

    train_dataset = LMDataset(corpus_path=args.corpus_path,
                              tokenizer=tokenizer,
                              local_rank=args.local_rank,
                              seq_len=args.max_seq_length,
                              vocab_size=args.vocab_size,
                              mask_prob=args.mask_prob)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    global_step = 0
    iters = 0
    model.zero_grad()
    model.train()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    while True:
        train_dataset.gen_segment()
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size,
                                      num_workers=4,
                                      pin_memory=True)
        epoch_iterator = tqdm(train_dataloader,
                              desc="Training (X iter) (XX / XX Steps) (Total Loss=X.X)\
                               (Generator Loss=X.X) (Discriminator Loss=X.X)",
                              disable=args.local_rank not in [-1, 0])
        tr_loss = 0.0
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids = batch
            gen_loss, disc_loss = model(input_ids, segment_ids, input_mask, lm_label_ids)

            loss = gen_loss + disc_loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            mean_loss = tr_loss * args.gradient_accumulation_steps / (step+1)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # learning rate warmup
                optimizer.step()
                for param in model.parameters():
                    param.grad = None
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d iter) (%d / %d Steps) (Mean Loss=%2.5f) (Generator Loss=%2.5f) (Discriminator Loss=%2.5f)"
                    % (iters, global_step, args.num_steps, mean_loss, gen_loss, disc_loss/50.0))

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('Mean_Loss', mean_loss, global_step)
                    tb_writer.add_scalar('Gen_Loss', gen_loss, global_step)
                    tb_writer.add_scalar('Disc_Loss', disc_loss/50.0, global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_checkpoint = os.path.join(args.output_dir, args.model_name+'_'+str(global_step)+'.bin')
                    model_layer_checkpoint = os.path.join(args.output_dir, args.model_name+'_'+str(global_step)+'_disc.bin')
                    torch.save(model_to_save.state_dict(), model_checkpoint)
                    torch.save(model_to_save.discriminator.model.state_dict(), model_layer_checkpoint)
                    logger.info("Saving model checkpoint to %s", args.output_dir)
            if args.num_steps > 0 and global_step == args.num_steps:
                epoch_iterator.close()
                break
        if args.num_steps > 0 and global_step == args.num_steps:
            epoch_iterator.close()
            break
        iters += 1
    if args.local_rank in [-1, 0]:
        model_to_save = model.module if hasattr(model, 'module') else model
        model_checkpoint = os.path.join(args.output_dir, args.model_name+'_'+str(global_step)+'.bin')
        model_layer_checkpoint = os.path.join(args.output_dir, args.model_name+'_'+str(global_step)+'_disc.bin')
        torch.save(model_to_save.state_dict(), model_checkpoint)
        torch.save(model_to_save.discriminator.model.state_dict(), model_layer_checkpoint)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        logger.info("End Training!")
        tb_writer.close()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--corpus_path", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument("--gen_config_file", type=str, required=True,
                        help="generator model configuration file")
    parser.add_argument("--disc_config_file", type=str, required=True,
                        help="discriminator model configuration file")

## Other parameters
    parser.add_argument("--output_dir", default='checkpoint', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_steps", default=1000000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Masking probability (Small, Base=0.15 Large=0.25)")

    parser.add_argument('--logging_steps', type=int, default=8,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0.0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    tokenizer = BertTokenizer(vocab_file='dataset/wiki_vocab_32k_0213.txt', do_lower_case=args.do_lower_case,
                              max_len=args.max_seq_length, do_basic_tokenize=True)
    generator_config = Config(args.gen_config_file)
    discriminator_config = Config(args.disc_config_file)
    model = Electra(generator_config, discriminator_config)

    gen_params = count_parameters(model.generator)
    disc_params = count_parameters(model.discriminator)

    logger.info("Generator Model Parameter: %d" % gen_params)
    logger.info("Discriminator Model Parameter: %d" % disc_params)

    logger.info("Generator Configuration: %s" % generator_config)
    logger.info("Discriminator Configuration: %s" % discriminator_config)
    model.to(args.device)

    args.vocab_size = generator_config.vocab_size
    logger.info("Training parameters %s", args)

    # Training
    train(args, model, tokenizer)


if __name__ == "__main__":
    main()
