import argparse
import json
import os
import sys
import time

__all__ = ["parse_parameters", "dump_json", "ensure_list", "pad_sequence_to_length"]

'''Update date: 2019-04-01'''
def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--dataset", default="")
    parser.add_argument("--train_filepath", default="")
    parser.add_argument("--dev_filepath", default="")
    parser.add_argument("--test_filepath", default="")
    parser.add_argument("--pretrained_word_embeddings", default="/data/dai031/Corpora/GloVe/glove.6B.100d.txt")
    parser.add_argument("--elmo_json", default="")
    parser.add_argument("--elmo_hdf5", default="")
    parser.add_argument("--num_instances", type=int, default=0)
    parser.add_argument("--use_pos", type=int, default=0)
    parser.add_argument("--max_vocab_size", type=int, default=0)

    # network
    parser.add_argument("--word_embedding_size", type=int, default=100)
    parser.add_argument("--char_embedding_size", type=int, default=16)
    parser.add_argument("--label_embedding_size", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lstm_cell_size", type=int, default=200)
    parser.add_argument("--lstm_layers", type=int, default=2)

    # train
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--validation_metric", default="")
    parser.add_argument("--gradient_norm", type=int, default=5)

    parser.add_argument("--output_dir", default="")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--max_save_models", type=int, default=1)

    args, _ = parser.parse_known_args()
    return vars(args)


'''Update date: 2019-04-01'''
def dump_json(file_path, dict):
    if os.path.exists(file_path):
        old_json = json.load(open(file_path))
        if type(old_json) != list:
            old_json = [old_json]
    else:
        old_json = []

    old_json.append(dict)

    with open(file_path, "w") as f:
        f.write(json.dumps(old_json, indent=2))


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py
Update date: 2019-03-01'''
def ensure_list(iterable):
    if isinstance(iterable, list):
        return iterable
    else:
        return list(iterable)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py
Update date: 2019-03-01'''
def pad_sequence_to_length(sequence, desired_length, default_value=lambda: 0):
    padded_sequence = sequence[:desired_length]
    for _ in range(desired_length - len(padded_sequence)):
        padded_sequence.append(default_value())
    return padded_sequence


'''Reference url: https://github.com/tensorflow/nmt/blob/tf-1.4/nmt/utils/misc_utils.py
Update date: 2019-April-29'''
def print_out(output, new_line=True):
    print(output, end="", file=sys.stdout)
    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


'''Reference url: https://github.com/tensorflow/nmt/blob/tf-1.4/nmt/utils/misc_utils.py
Update date: 2019-April-29'''
def print_time(output, start_time):
    print("%s, time %ds, %s." % (output, (time.time() - start_time), time.ctime()))
    sys.stdout.flush()
    return time.time()
