import numpy as np
import collections


def _is_start_piece(piece):
    special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
    piece = ''.join(piece)
    if (piece.startswith("▁") or piece.startswith("<")
        or piece in special_pieces):
        return True
    else:
        return False


def _sample_mask(seg, mask_alpha, tokenizer, mask_beta,
                 max_gram=3, goal_num_predict=80):
    # try to n-gram masking SpanBERT(Joshi et al., 2019)
    # 3-gram implementation
    seg_len = len(seg)
    mask = np.array([False] * seg_len, dtype=np.bool)

    num_predict = 0

    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_gram + 1)
    pvals /= pvals.sum(keepdims=True) # p(n) = 1/n / sigma(1/k)

    cur_len = 0

    while cur_len < seg_len:
        if goal_num_predict is not None and num_predict >= goal_num_predict: break

        n = np.random.choice(ngrams, p=pvals)
        if goal_num_predict is not None:
            n = min(n, goal_num_predict - num_predict)

        # `mask_alpha` : number of tokens forming group
        # `mask_beta` : number of tokens to be masked in each groups.
        ctx_size = (n * mask_alpha) // mask_beta
        l_ctx = np.random.choice(ctx_size)
        r_ctx = ctx_size - l_ctx

        # Find the start position of a complete token
        beg = cur_len + l_ctx

        while beg < seg_len and not _is_start_piece([seg[beg]]):
            beg += 1
        if beg >= seg_len:
            break

        # Find the end position of the n-gram (start pos of the n+1-th gram)
        end = beg + 1
        cnt_ngram = 1
        while end < seg_len:
            if _is_start_piece([seg[beg]]):
                cnt_ngram += 1
                if cnt_ngram > n:
                    break
            end += 1
        if end >= seg_len:
            break

        # Update
        mask[beg:end] = True
        num_predict += end - beg

        cur_len = end + r_ctx

    while goal_num_predict is not None and num_predict < goal_num_predict:
        i = np.random.randint(seg_len)
        if not mask[i]:
            mask[i] = True
            num_predict += 1

    tokens, masked_lm_labels = [], []
    for i in range(seg_len):
        if mask[i] and (seg[i] != '[CLS]' and seg[i] != '[SEP]'):
            masked_lm_labels.extend(tokenizer.convert_tokens_to_ids([seg[i]]))
            tokens.append('[MASK]')
        else:
            tokens.append(seg[i])
            masked_lm_labels.append(-1)
    return tokens, masked_lm_labels
