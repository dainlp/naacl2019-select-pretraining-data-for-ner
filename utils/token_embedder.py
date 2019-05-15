import math
import numpy as np
import os
import torch
from typing import Dict

from .elmo import Elmo
from .seq2vec import CnnEncoder
from .time_distributed import TimeDistributed


'''Update at April-17-2019'''
class _TokenEmbedder(torch.nn.Module):
    def get_output_dim(self):
        raise NotImplementedError


'''Update at April-17-2019'''
class Embedding(_TokenEmbedder):
    def __init__(self, vocab_size, embedding_size, weight: torch.FloatTensor = None,
                 trainable=True, scale_by_embedding_size=False):
        super(Embedding, self).__init__()
        self.output_dim = embedding_size
        self.scale_by_embedding_size = scale_by_embedding_size

        if weight is None:
            weight = torch.FloatTensor(vocab_size, embedding_size)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            assert weight.size() == (vocab_size, embedding_size)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs):
        outs = torch.nn.functional.embedding(inputs, self.weight)
        if self.scale_by_embedding_size:
            outs = outs * math.sqrt(self.output_dim)
        return outs


'''Update at April-17-2019'''
class TokenCharactersEmbedder(_TokenEmbedder):
    def __init__(self, embedding: Embedding, encoder, dropout=0.0):
        super(TokenCharactersEmbedder, self).__init__()
        self._embedding = TimeDistributed(embedding)
        self._encoder = TimeDistributed(encoder)
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    def get_output_dim(self):
        return self._encoder._module.get_output_dim()

    def forward(self, token_characters):
        '''token_characters: batch_size, num_tokens, num_characters'''
        mask = (token_characters != 0).long()
        outs = self._embedding(token_characters)
        outs = self._encoder(outs, mask)
        outs = self._dropout(outs)
        return outs


'''Update at April-17-2019'''
class ElmoTokenEmbedder(_TokenEmbedder):
    def __init__(self, options_file, weight_file, do_layer_norm=False, dropout=0.5, requires_grad=False):
        super(ElmoTokenEmbedder, self).__init__()

        self._elmo = Elmo(options_file, weight_file, num_output_representations=1)
        self.output_dim = self._elmo.get_output_dim()

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs):
        '''inputs: batch_size, num_tokens, 50'''
        elmo_output = self._elmo(inputs)
        return elmo_output["elmo_representations"][0]


'''Update at April-20-2019'''
def _load_pretrained_embeddings(filepath, dimension, token2idx):
    tokens_to_keep = set(token2idx.keys()) # TODO: if token is uppercase and pretrained is lowercase
    embeddings = {}
    if filepath != "" and os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                sp = line.strip().split(" ")
                if len(sp) <= dimension: continue
                token = sp[0]
                if token not in tokens_to_keep: continue
                embeddings[token] = np.array([float(x) for x in sp[1:]])

    print(" # Load %d out of %d words (%d-dimensional) from pretrained embedding file (%s)!" % (
    len(embeddings), len(token2idx), dimension, filepath))

    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    weights = np.random.normal(embeddings_mean, embeddings_std, size=(len(token2idx), dimension))
    for token, i in token2idx.items():
        if token in embeddings:
            weights[i] = embeddings[token]
    return weights


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/text_field_embedders/*
Update at April-20-2019
Takes as input the dict produced by TextField and 
returns as output an embedded representations of the tokens in that field'''
class TextFieldEmbedder(torch.nn.Module):
    def __init__(self, embedders):
        super(TextFieldEmbedder, self).__init__()

        self.embedders = embedders
        for k, embedder in embedders.items():
            self.add_module("embedder_%s" % k, embedder)

    def get_output_dim(self):
        dim = [embedder.get_output_dim() for embedder in self.embedders.values()]
        return sum(dim)

    '''Each tensor in here is assumed to have a shape roughly similar to (batch_size, num_tokens)'''
    def forward(self, text_field_input: Dict[str, torch.Tensor]):
        assert self.embedders.keys() == text_field_input.keys()

        outs = []
        for k in sorted(self.embedders.keys()):
            tensors = [text_field_input[k]]
            embedder = getattr(self, "embedder_%s" % k)
            outs.append(embedder(*tensors))
        return torch.cat(outs, dim=-1)

    @classmethod
    def create_embedder(cls, vocab, args):
        embedders = {}

        token2idx = vocab.get_item_to_index_vocabulary("tokens")
        weight = _load_pretrained_embeddings(args["pretrained_word_embeddings"], args["word_embedding_size"], token2idx)
        embedders["tokens"] = Embedding(len(token2idx), args["word_embedding_size"], weight=torch.FloatTensor(weight))

        embedding = Embedding(vocab.get_vocab_size("token_characters"), args["char_embedding_size"])
        embedders["token_characters"] = TokenCharactersEmbedder(embedding, CnnEncoder())

        if args["use_pos"] == 1:
            embedders["pos"] = Embedding(vocab.get_vocab_size("pos"), args["char_embedding_size"])

        if args["elmo_json"] != "" and args["elmo_hdf5"] != "":
            embedders["elmo_characters"] = ElmoTokenEmbedder(args["elmo_json"], args["elmo_hdf5"])

        return cls(embedders)