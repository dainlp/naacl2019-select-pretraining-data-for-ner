import torch
from typing import Dict, List, Iterable

from .common import pad_sequence_to_length
from .pytorch import batch_tensor_dicts
from .tooken import Token


__all__ = ["MetadataField", "TextField", "SequenceLabelField", "Instance"]


'''Update April-17-2019'''
class MetadataField:
    def __init__(self, metadata):
        self.metadata = metadata

    def count_vocab_items(self, counter):
        pass

    def index(self, vocab):
        pass

    def get_padding_lengths(self):
        return {}

    def as_tensor(self, padding_lengths):
        return self.metadata

    @classmethod
    def batch_tensors(self, tensor_list):
        return tensor_list


'''Update April-14-2019'''
class TextField:
    def __init__(self, tokens: List[Token], token_indexers):
        self.tokens = tokens
        self._token_indexers = token_indexers
        self._indexed_tokens = None

    def count_vocab_items(self, counter):
        for indexer in self._token_indexers.values():
            for token in self.tokens:
                indexer.count_vocab_items(token, counter)

    def index(self, vocab):
        token_arrays = {}
        for indexer_name, indexer in self._token_indexers.items():
            token_indices = indexer.tokens_to_indices(self.tokens, vocab, indexer_name)
            token_arrays.update(token_indices)
        self._indexed_tokens = token_arrays

    def get_padding_lengths(self):
        lengths = []

        for indexer_name, indexer in self._token_indexers.items():
            indexer_lengths = {}
            token_lengths = [indexer.get_padding_lengths(token) for token in self._indexed_tokens[indexer_name]]
            for key in token_lengths[0]:
                indexer_lengths[key] = max(x[key] if key in x else 0 for x in token_lengths)
            lengths.append(indexer_lengths)

        padding_lengths = {}
        for indexer_name, token_list in self._indexed_tokens.items():
            padding_lengths[f"{indexer_name}_length"] = len(token_list)

        padding_keys = {key for d in lengths for key in d.keys()}
        for padding_key in padding_keys:
            padding_lengths[padding_key] = max(x[padding_key] if padding_key in x else 0 for x in lengths)

        return padding_lengths

    def sequence_length(self):
        return len(self.tokens)

    def as_tensor(self, padding_lengths):
        tensors = {}
        for indexer_name, indexer in self._token_indexers.items():
            desired_num_tokens = {indexer_name: padding_lengths[f"{indexer_name}_length"]}
            indices_to_pad = {indexer_name: self._indexed_tokens[indexer_name]}
            padded_array = indexer.pad_token_sequence(indices_to_pad, desired_num_tokens, padding_lengths)
            indexer_tensors = {key: torch.LongTensor(array) for key, array in padded_array.items()}
            tensors.update(indexer_tensors)
        return tensors

    def batch_tensors(self, tensor_dicts):
        return batch_tensor_dicts(tensor_dicts)

    '''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py (get_text_field_mask)
        Update date: May-01-2019'''

    @classmethod
    def get_text_field_mask(cls, text_field_tensors: Dict[str, torch.Tensor]):
        if "mask" in text_field_tensors:
            return text_field_tensors["mask"]

        tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
        tensor_dims.sort(key=lambda x: x[0])

        assert tensor_dims[0][0] == 2

        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/fields/label_field.py
Update at May-01-2019'''
class LabelField:
    def __init__(self, label, skip_indexing=False):
        self._key = "labels"
        self.label = label
        self._label_id = None

        if skip_indexing: self._label_id = label

    def count_vocab_items(self, counter):
        if self._label_id is None:
            counter[self._key][self.label] += 1

    def index(self, vocab):
        if self._label_id is None:
            self._label_id = vocab.get_item_index(self.label, self._key)

    def get_padding_lengths(self):
        return {}

    def as_tensor(self, padding_lengths):
        return torch.tensor(self._label_id, dtype=torch.long)

    def batch_tensors(self, tensor_list):
        return torch.stack(tensor_list)


'''Update at April-14-2019'''
class SequenceLabelField:
    def __init__(self, labels, inputs):
        self._key = "labels"
        self.labels = labels
        self._indexed_labels = None
        self.inputs = inputs

        if all([isinstance(l, int) for l in labels]):
            self._indexed_labels = labels

    def count_vocab_items(self, counter):
        if self._indexed_labels is None:
            for label in self.labels:
                counter[self._key][label] += 1

    def index(self, vocab):
        if self._indexed_labels is None:
            self._indexed_labels = [vocab.get_item_index(label, self._key) for label in self.labels]

    def get_padding_lengths(self):
        return {"num_tokens": self.inputs.sequence_length()}

    def as_tensor(self, padding_lengths):
        desired_num_tokens = padding_lengths["num_tokens"]
        padded_labels = pad_sequence_to_length(self._indexed_labels, desired_num_tokens)
        return torch.LongTensor(padded_labels)

    def batch_tensors(self, tensor_list):
        return torch.stack(tensor_list)


'''Update at April-22-2019'''
class ActionField:
    def __init__(self, actions, inputs):
        self._key = "actions"
        self.actions = actions
        self._indexed_actions = None
        self.inputs = inputs

        if all([isinstance(a, int) for a in actions]):
            self._indexed_actions = actions

    def count_vocab_items(self, counter):
        if self._indexed_actions is None:
            for action in self.actions:
                counter[self._key][action] += 1

    def index(self, vocab):
        if self._indexed_actions is None:
            self._indexed_actions = [vocab.get_item_index(action, self._key) for action in self.actions]

    def get_padding_lengths(self):
        return {"num_tokens": self.inputs.sequence_length() * 2}

    def as_tensor(self, padding_lengths):
        desired_num_actions = padding_lengths["num_tokens"]
        padded_actions = pad_sequence_to_length(self._indexed_actions, desired_num_actions)
        return torch.LongTensor(padded_actions)

    def batch_tensors(self, tensor_list):
        return torch.stack(tensor_list)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py
Update at May-4-2019'''
class LazyInstances(Iterable):
    def __init__(self, instance_generator):
        super().__init__()
        self.instance_generator = instance_generator

    def __iter__(self):
        instances = self.instance_generator()
        yield from instances

    
'''Update at April-14-2019'''
class Instance:
    def __init__(self, fields):
        self.fields = fields
        self.indexed = False

    def count_vocab_items(self, counter):
        for field in self.fields.values():
            field.count_vocab_items(counter)

    def index_fields(self, vocab):
        if not self.indexed:
            self.indexed = True
            for field in self.fields.values():
                field.index(vocab)

    def get_padding_lengths(self):
        lengths = {}
        for field_name, field in self.fields.items():
            lengths[field_name] = field.get_padding_lengths()
        return lengths

    def as_tensor_dict(self, padding_lengths):
        padding_lengths = padding_lengths or self.get_padding_lengths()
        tensors = {}
        for field_name, field in self.fields.items():
            tensors[field_name] = field.as_tensor(padding_lengths[field_name])
        return tensors
