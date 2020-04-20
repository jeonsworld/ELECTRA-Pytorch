import logging
import torch
import os
import pickle

from random import randint, shuffle

from torch.utils.data import Dataset
from tqdm import tqdm
from utils.masking_utils import _sample_mask

logger = logging.getLogger(__name__)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        if len(tokens_a) + len(tokens_b) <= max_num_tokens:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_instances(documents, max_len, local_rank, short_seq_prob=0.1):
    instances = []
    current_chunk = []
    current_length = 0
    max_len = max_len - 3  # [CLS] [SEP] * 2
    for i, doc in enumerate(tqdm(documents, desc="Create Instances", unit=" doc", disable=local_rank not in [-1, 0])):
        seq_len = max_len
        for d in doc:
            current_chunk.append(d)
            current_length += len(d)
            if current_length >= seq_len:
                if current_chunk and len(current_chunk) >= 2:
                    a_end = randint(1, len(current_chunk)-1)

                    tokens_a = []
                    for index in range(a_end):
                        tokens_a.extend(current_chunk[index])

                    tokens_b = []
                    for index in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[index])

                    instance = (tokens_a, tokens_b)
                    instances.append(instance)
                    current_chunk = []
                    current_length = 0

        if current_chunk and len(current_chunk) >= 2:
            a_end = randint(1, len(current_chunk)-1)

            tokens_a = []
            for index in range(a_end):
                tokens_a.extend(current_chunk[index])

            tokens_b = []
            for index in range(a_end, len(current_chunk)):
                tokens_b.extend(current_chunk[index])
            instance = (tokens_a, tokens_b)
            instances.append(instance)
        current_chunk = []
        current_length = 0

    return instances


class LMDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, local_rank, seq_len=512, vocab_size=32000, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.max_len = seq_len
        self.docs = []
        self.mask_prob = mask_prob
        self.max_predictions_per_seq = 80
        self.vocab_size = vocab_size
        self.local_rank = local_rank
        # self.epoch = 0
        doc = []
        new_line_check = 0
        num_line = 0
        logger.info("LMDataset init...")
        cached_features_file = corpus_path + '.cache'
        if os.path.exists(cached_features_file):
            logger.info("Loading dataset from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as reader:
                self.docs = pickle.load(reader)
        else:
            with open(corpus_path, 'r', encoding='utf-8')as f:
                for line in tqdm(f, desc="Loading Dataset", unit=" lines", disable=local_rank not in [-1, 0]):
                    line = line.strip()
                    if line == "" or line == "":
                        if num_line <= 1 and new_line_check == 0:
                            doc = []
                            new_line_check += 1
                            num_line = 0
                        elif new_line_check == 0:
                            self.docs.append(doc)
                            doc = []
                            new_line_check += 1
                            num_line = 0
                        else:
                            continue
                    else:
                        tokens = tokenizer.tokenize(line)
                        doc.append(tokens)
                        new_line_check = 0
                        num_line += 1

                if doc:
                    self.docs.append(doc)  # If the last doc didn't end on a newline, make sure it still gets added

                if len(self.docs) <= 1:
                    exit(1)

                logger.info("Saving dataset into cached file %s", cached_features_file)
                with open(cached_features_file, "wb") as writer:
                    pickle.dump(self.docs, writer, protocol=pickle.HIGHEST_PROTOCOL)

    def gen_segment(self):
        shuffle(self.docs)
        self.instance = create_instances(self.docs, self.max_len, self.local_rank)

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, item):
        instance = self.instance[item]

        tokens_a = instance[0]
        tokens_b = instance[1]

        truncate_seq_pair(tokens_a, tokens_b, self.max_len-3)
        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]
        input_mask = [1]*len(tokens)

        n_prob = min(self.max_predictions_per_seq, max(1, int(round(len(tokens) * self.mask_prob))))
        tokens, masked_lm_labels = _sample_mask(seg=tokens, tokenizer=self.tokenizer, mask_alpha=4,
                                                mask_beta=1, max_gram=3, goal_num_predict=n_prob)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)
        masked_lm_labels.extend([-1]*n_pad)
        tensors = (torch.tensor(input_ids, dtype=torch.long),
                   torch.tensor(input_mask, dtype=torch.long),
                   torch.tensor(segment_ids, dtype=torch.long),
                   torch.tensor(masked_lm_labels, dtype=torch.long))
        return tensors
