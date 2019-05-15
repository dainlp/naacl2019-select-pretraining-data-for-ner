from collections import defaultdict
import itertools
import math
import random
from typing import List, Tuple, cast, Dict

from .common import ensure_list


__all__ = ["Batch", "BasicIterator", "BucketIterator"]


'''Update at April-11-2019'''
class Batch:
    def __init__(self, instances):
        super().__init__()
        self.instances = ensure_list(instances)

    '''return: {'tokens': {'tokens_length': 45, 'token_characters_length': 45, 'elmo_characters_length': 45, 
                        'num_token_characters': 15}, 'tags': {'num_tokens': 45}})'''
    def get_padding_lengths(self):
        padding_lengths = defaultdict(dict)
        all_instance_lengths = [instance.get_padding_lengths() for instance in self.instances]

        all_field_lengths = defaultdict(list)
        for instance_lengths in all_instance_lengths:
            for field_name, instance_field_lengths in instance_lengths.items():
                all_field_lengths[field_name].append(instance_field_lengths)

        for field_name, field_lengths in all_field_lengths.items():
            for padding_key in field_lengths[0].keys():
                max_value = max(x[padding_key] if padding_key in x else 0 for x in field_lengths)
                padding_lengths[field_name][padding_key] = max_value

        return padding_lengths

    def as_tensor_dict(self, padding_lengths=None):
        if padding_lengths is None:
            padding_lengths = defaultdict(dict)

        instance_padding_lengths = self.get_padding_lengths()

        lengths_to_use = defaultdict(dict)
        for field_name, instance_field_lengths in instance_padding_lengths.items():
            for padding_key in instance_field_lengths.keys():
                if padding_lengths[field_name].get(padding_key) is not None:
                    lengths_to_use[field_name][padding_key] = padding_lengths[field_name][padding_key]
                else:
                    lengths_to_use[field_name][padding_key] = instance_field_lengths[padding_key]


        field_tensors = defaultdict(list)

        for instance in self.instances:
            for field, tensors in instance.as_tensor_dict(lengths_to_use).items():
                field_tensors[field].append(tensors)

        field_classes = self.instances[0].fields

        final_fields = {}
        for field_name, field_tensor_list in field_tensors.items():
            final_fields[field_name] = field_classes[field_name].batch_tensors(field_tensor_list)
        return final_fields

    def __iter__(self):
        return iter(self.instances)

    def index_instances(self, vocab):
        for instance in self.instances:
            instance.index_fields(vocab)


'''Update at April-17-2019'''
class _Iterator:
    def __init__(self, batch_size=64):
        self.vocab = None
        self._batch_size = batch_size

    def __call__(self, instances, shuffle=True):
        batches = self._create_batches(instances, shuffle)
        for batch in batches:
            if self.vocab is not None:
                batch.index_instances(self.vocab)
            padding_lengths = batch.get_padding_lengths()
            tensor_dict = batch.as_tensor_dict(padding_lengths)
            yield tensor_dict

    def get_num_batches(self, instances):
        return math.ceil(len(instances) / self._batch_size)

    def index_with(self, vocab):
        self.vocab = vocab

    @classmethod
    def lazy_group_of(cls, iterator, group_size):
        return iter(lambda: list(itertools.islice(iterator, 0, group_size)), [])


'''Update at April-16-2019'''
class BasicIterator(_Iterator):
    def _create_batches(self, instances, shuffle=True):
        if shuffle:
            random.shuffle(instances)
        for batch_instances in _Iterator.lazy_group_of(iter(instances), self._batch_size):
            yield Batch(batch_instances)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py
Update at April-17-2019'''
def _add_noise_to_dict_values(dictionary, noise_param):
    dict_with_noise = {}
    for key, value in dictionary.items():
        noise_value = value * noise_param
        noise = random.uniform(-noise_value, noise_value)
        dict_with_noise[key] = value + noise
    return dict_with_noise


'''Update at April-17-2019'''
def _sort_by_padding(instances, sorting_keys: List[Tuple[str, str]], vocab, padding_noise=0.1):
    instances_with_lengths = []
    for instance in instances:
        instance.index_fields(vocab)
        padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())

        if padding_noise > 0.0:
            noisy_lengths = {}
            for field_name, field_lengths in padding_lengths.items():
                noisy_lengths[field_name] = _add_noise_to_dict_values(field_lengths, padding_noise)
            padding_lengths = noisy_lengths

        instance_with_lengths = ([padding_lengths[field_name][padding_key]
                                  for (field_name, padding_key) in sorting_keys], instance)
        instances_with_lengths.append(instance_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0])
    sorted_instances = [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]
    return sorted_instances


'''Update at April-17-2019'''
class BucketIterator(_Iterator):
    def __init__(self, sorting_keys: List[Tuple[str, str]], padding_noise=0.1, batch_size=64):
        super(BucketIterator, self).__init__(batch_size=batch_size)
        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise

    def _create_batches(self, instances, shuffle=True):
        instances = _sort_by_padding(instances, self._sorting_keys, self.vocab, self._padding_noise)

        batches = []
        for batch_instances in _Iterator.lazy_group_of(iter(instances), self._batch_size):
            batches.append(Batch(batch_instances))

        if shuffle:
            random.shuffle(batches)

        yield from batches


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py
Update at May-4-2019'''
class DatasetReader:
    def __init__(self, lazy=False):
        self.lazy = lazy