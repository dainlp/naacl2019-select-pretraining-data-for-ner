import os, sys
sys.path.insert(0, os.path.abspath("../.."))

import shutil
import torch
print(torch.__version__)

from utils.common import dump_json, parse_parameters, print_out
from utils.crf import CrfTagger
from utils.data import BasicIterator, BucketIterator
from utils.instance import MetadataField, TextField, SequenceLabelField, Instance
from utils.metrics import compute_f1
from utils.ner import to_bioes
from utils.train import create_output_dir, set_random_seed, train, final_evaluate, generate_predictions
from utils.tooken import Token
from utils.token_indexer import SingleIdTokenIndexer, TokenCharactersIndexer, ELMoTokenCharactersIndexer
from utils.vocab import Vocabulary


def _update_default_parameters(args):
    args["learning_rate"] = 0.001
    args["validation_metric"] = "f1-measure-overall"
    return args


class DatasetReader:
    def __init__(self, args):
        self._token_indexers = {"tokens": SingleIdTokenIndexer(),
                                "token_characters": TokenCharactersIndexer()}
        if args["elmo_json"] != "" and args["elmo_hdf5"] != "":
            self._token_indexers["elmo_characters"] = ELMoTokenCharactersIndexer()

    def read(self, filepath):
        instances = []
        with open(filepath, "r") as f:
            for raw_text in f:
                tokens = [Token(t) for t in raw_text.strip().split()]
                tags = ["O"] * len(tokens)
                raw_mentions = next(f).strip()
                if len(raw_mentions) > 0:
                    for mention in raw_mentions.split("|"):
                        indices, label = mention.split(" ")
                        indices = [int(i) for i in indices.split(",")]
                        assert len(indices) == 2 and indices[0] <= indices[1]
                        tags[indices[0]] = "B-%s" % label
                        for i in range(indices[0] + 1, indices[1] + 1):
                            tags[i] = "I-%s" % label
                instances.append(self._to_instance(raw_text, raw_mentions, tokens, tags))

                assert next(f).strip() == ""
        return instances

    def _to_instance(self, raw_text, raw_mentions, tokens, tags):
        text_fields = TextField(tokens, self._token_indexers)
        sequence_label_fields = SequenceLabelField(to_bioes(tags), text_fields)
        return Instance({"raw_text": MetadataField(raw_text.strip()),
                         "raw_mentions": MetadataField(raw_mentions.strip()),
                         "tokens": text_fields, "tags": sequence_label_fields})


def train_model(args):
    model = CrfTagger(vocab, args).cuda(0)
    parameters = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args["learning_rate"])

    metrics, model_paths = train(model, optimizer, train_data, train_iterator,
                                 dev_data, dev_iterator, args)
    return model, metrics, model_paths


def _external_eval(model, vocab, data, iterator):
    predictions = generate_predictions(model, data, iterator)
    gold_corpus = [str(instance.fields["raw_mentions"].metadata) for instance in data]
    pred_corpus = [instance["pred"] for instance in predictions]
    return compute_f1(gold_corpus, pred_corpus)


if __name__ == "__main__":
    args = parse_parameters()
    args = _update_default_parameters(args)

    set_random_seed(seed=args["random_seed"])
    if os.path.exists(args["output_dir"]):
        shutil.rmtree(args["output_dir"])
    create_output_dir(args["output_dir"])

    dataset_reader = DatasetReader(args)
    train_data = dataset_reader.read("data/%s/train.txt" % args["dataset"])
    print_out("Load %d instances from train set." % (len(train_data)))
    dev_data = dataset_reader.read("data/%s/dev.txt" % args["dataset"])
    print_out("Load %d instances from dev set." % (len(dev_data)))
    test_data = dataset_reader.read("data/%s/test.txt" % args["dataset"])
    print_out("Load %d instances from test set." % (len(test_data)))

    datasets = {"train": train_data, "validation": dev_data, "test": test_data}
    vocab = Vocabulary.from_instances((instance for dataset in datasets.values() for instance in dataset))
    vocab.save_to_files(os.path.join(args["output_dir"], "vocabulary"))
    train_iterator = BucketIterator(sorting_keys=[['tokens', 'tokens_length']], batch_size=args["batch_size"])
    train_iterator.index_with(vocab)
    dev_iterator = BasicIterator(batch_size=args["batch_size"])
    dev_iterator.index_with(vocab)

    model, metrics, model_paths = train_model(args)
    metrics["args"] = args

    final_evaluate(model, vocab, test_data, dev_iterator, _external_eval, metrics, args, model_paths)
