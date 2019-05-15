'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/vocabulary.py
Update date: April-16-2019'''
import codecs
from collections import defaultdict
import os

__all__ = ["Vocabulary"]

class _NamespaceDependentDefaultDict(defaultdict):
    def __init__(self, padded_function, non_padded_function):
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super(_NamespaceDependentDefaultDict, self).__init__()


    def __missing__(self, key):
        if key.endswith("labels") or key.endswith("tags"):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, padding_item, oov_item):
        super(_TokenToIndexDefaultDict, self).__init__(lambda: {padding_item: 0, oov_item: 1},
                                                       lambda: {})


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, padding_item, oov_item):
        super(_IndexToTokenDefaultDict, self).__init__(lambda: {0: padding_item, 1: oov_item},
                                                       lambda: {})


class Vocabulary:
    def __init__(self, counter, max_vocab_size=0):
        self._padding_item = "@@PADDING@@"
        self._oov_item = "@@UNKNOWN@@"
        self._item_to_index = _TokenToIndexDefaultDict(self._padding_item, self._oov_item)
        self._index_to_item = _IndexToTokenDefaultDict(self._padding_item, self._oov_item)
        self._extend(counter=counter, max_vocab_size=max_vocab_size)


    def save_to_files(self, directory):
        os.makedirs(directory, exist_ok=True)

        for namespace, mapping in self._index_to_item.items():
            with codecs.open(os.path.join(directory, namespace + ".txt"), "w", "utf-8") as f:
                for i in range(len(mapping)):
                    print(mapping[i].replace("\n", "@@NEWLINE@@"), file=f)


    @classmethod
    def from_instances(cls, instances, max_vocab_size=0):
        counter = defaultdict(lambda: defaultdict(int))
        for instance in instances:
            instance.count_vocab_items(counter)
        return cls(counter=counter, max_vocab_size=max_vocab_size)


    def _extend(self, counter, max_vocab_size):
        for namespace in counter:
            item_counts = list(counter[namespace].items())
            item_counts.sort(key=lambda x: x[1], reverse=True)
            if max_vocab_size > 0: item_counts=item_counts[:max_vocab_size]
            for item, count in item_counts:
                self._add_item_to_namespace(item, namespace)


    def _add_item_to_namespace(self, item, namespace="tokens"):
        if item not in self._item_to_index[namespace]:
            idx = len(self._item_to_index[namespace])
            self._item_to_index[namespace][item] = idx
            self._index_to_item[namespace][idx] = item


    def get_index_to_item_vocabulary(self, namespace="tokens"):
        return self._index_to_item[namespace]


    def get_item_to_index_vocabulary(self, namespace="tokens"):
        return self._item_to_index[namespace]


    def get_item_index(self, item, namespace="tokens"):
        if item in self._item_to_index[namespace]:
            return self._item_to_index[namespace][item]
        else:
            return self._item_to_index[namespace][self._oov_item]


    def get_item(self, idx, namespace="tokens"):
        return self._index_to_item[namespace][int(idx)]


    def get_vocab_size(self, namespace="tokens"):
        return len(self._item_to_index[namespace])