# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn import CrossEntropyLoss, Dropout, Embedding, BCEWithLogitsLoss
from torch.autograd import Variable
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)


try:
    import apex
    #apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
    import apex.normalization
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
    #apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
    #BertLayerNorm = apex.normalization.FusedLayerNorm
    APEX_IS_AVAILABLE = True
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    #BertLayerNorm = BertNonFusedLayerNorm
    APEX_IS_AVAILABLE = False
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.shape = torch.Size((hidden_size,))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.apex_enabled = APEX_IS_AVAILABLE

    @torch.jit.unused
    def fused_layer_norm(self, x):
        return FusedLayerNormAffineFunction.apply(
            x, self.weight, self.bias, self.shape, self.eps)

    def forward(self, x):
        if self.apex_enabled and not torch.jit.is_scripting():
            x = self.fused_layer_norm(x)
        else:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight * x + self.bias
        return x


LayerNorm = BertLayerNorm


def Linear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "bias_gelu": bias_gelu, "bias_tanh": bias_tanh, "relu": torch.nn.functional.relu, "swish": swish}


class LinearActivation(nn.Module):
    r"""Fused Linear and activation Module.
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, act='gelu', bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_fn = nn.Identity()                                                         #
        self.biased_act_fn = None                                                           #
        self.bias = None                                                                    #
        if isinstance(act, str) or (sys.version_info[0] == 2 and isinstance(act, unicode)): # For TorchScript
            if bias and not 'bias' in act:                                                  # compatibility
                act = 'bias_' + act                                                         #
                self.biased_act_fn = ACT2FN[act]                                            #

            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if not self.bias is None:
            return self.biased_act_fn(self.bias, F.linear(input, self.weight, None))
        else:
            return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Config(object):
    def __init__(self,
                 vocab_size_or_config_json_file,
                 act_fn="gelu",
                 hidden_size=768,
                 embedding_size=768,
                 num_hidden_layers=12,
                 num_heads=12,
                 dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 disc_weight=50.0,
                 initializer_range=0.02,
                 generator_decay=4
                 ):
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.act_fn = act_fn
            self.hidden_size = hidden_size
            self.embedding_size = embedding_size
            self.num_hidden_layers = num_hidden_layers
            self.num_heads = num_heads
            self.dropout_prob = dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.disc_weight = disc_weight
            self.initializer_range = initializer_range
            self.generator_decay = generator_decay
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        config = Config(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Embeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.embedding_size)
        self.LayerNorm = LayerNorm(config.embedding_size, eps=1e-12)
        self.dropout = Dropout(config.dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / config.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.o_proj = Linear(config.hidden_size, config.hidden_size)
        self.dropout = Dropout(config.dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.o_proj(context_layer)
        return attention_output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = LinearActivation(config.hidden_size, config.hidden_size*4)
        self.fc2 = Linear(config.hidden_size*4, config.hidden_size)

    def forward(self, input):
        intermediate = self.fc1(input)
        ff_out = self.fc2(intermediate)
        return ff_out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feedforward = PositionWiseFeedForward(config)
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-12)
        self.feedforward_norm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.dropout_prob)

    def forward(self, hidden_states, attention_mask):
        # Multi-Head Self-Attention
        h = hidden_states
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + h
        hidden_states = self.attention_norm(hidden_states)
        # Position-Wise FeedForward
        h = hidden_states
        hidden_states = self.feedforward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + h
        hidden_states = self.feedforward_norm(hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layer.append(copy.deepcopy(Block(config)))

    def forward(self, hidden_states, attention_mask):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states, attention_mask)
        return hidden_states


class ElectraModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        #extended_attention_mask = extended_attention_mask.to(torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        if hasattr(self, "embeddings_project"):
            embedding_output = self.embeddings_project(embedding_output)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask)

        return encoded_layers


class ElectraGeneratorPredictionHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = Linear(config.hidden_size, config.embedding_size)
        self.dense = Linear(config.embedding_size, config.vocab_size, bias=False)
        self.bias = Parameter(torch.zeros(config.vocab_size))

        self.layer_norm = LayerNorm(config.embedding_size)
        self.act_fn = ACT2FN[config.act_fn]

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states) + self.bias
        return hidden_states


class ElectraGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.hidden_size //= config.generator_decay
        config.num_heads //= config.generator_decay
        self.model = ElectraModel(config)
        self.predictions = ElectraGeneratorPredictionHeads(config)

        self.vocab_size = config.vocab_size

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output = self.model(input_ids, token_type_ids, attention_mask)
        prediction_scores = self.predictions(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss, prediction_scores
        else:
            return prediction_scores


class ElectraDiscriminatorPredictionHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = LinearActivation(config.hidden_size, config.hidden_size)
        self.dense_prediction = Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dense_prediction(hidden_states)
        return hidden_states


class ElectraDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ElectraModel(config)
        self.predictions = ElectraDiscriminatorPredictionHeads(config)
        self.discriminator_weight = config.disc_weight

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output = self.model(input_ids, token_type_ids, attention_mask)
        prediction_scores = self.predictions(sequence_output).squeeze(-1)
        if masked_lm_labels is not None:
            loss_fct = BCEWithLogitsLoss()
            active_loss = attention_mask.view(-1, sequence_output.shape[1]) == 1
            active_logits = prediction_scores.view(-1, sequence_output.shape[1])[active_loss]
            active_labels = masked_lm_labels[active_loss]
            masked_lm_loss = loss_fct(active_logits, active_labels.float())
            masked_lm_loss *= self.discriminator_weight
            return masked_lm_loss
        else:
            return prediction_scores


class Electra(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.generator = ElectraGenerator(config)
        self.discriminator = ElectraDiscriminator(config)
        self._tie_embeddings()
        self.apply(self._init_weights)

    def _tie_embeddings(self):
        self.generator.model.embeddings.word_embeddings.weight = self.discriminator.model.embeddings.word_embeddings.weight
        self.generator.model.embeddings.word_embeddings.weight = self.generator.predictions.dense.weight

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _sampling(self, input_ids, generator_logits, masked_lm_labels):
        generator_id = torch.argmax(generator_logits, dim=-1).detach()
        origin_input = input_ids.clone()
        fake_input = input_ids.clone()
        fake_input = torch.where(masked_lm_labels < 0, fake_input, generator_id)
        corrupt_label = (masked_lm_labels != -1)
        origin_input[corrupt_label] = masked_lm_labels[corrupt_label]
        discriminator_label = torch.eq(origin_input, fake_input)
        return generator_id, discriminator_label

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        generator_loss, generator_logits = self.generator(input_ids, token_type_ids, attention_mask, masked_lm_labels)
        discriminator_input, discriminator_label = self._sampling(input_ids,
                                                                  generator_logits,
                                                                  masked_lm_labels)
        disc_loss = self.discriminator(discriminator_input, token_type_ids, attention_mask, discriminator_label)
        return generator_loss, disc_loss


class QuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ElectraModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = Linear(config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output = self.model(input_ids, token_type_ids, attention_mask)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
