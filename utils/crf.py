from collections import defaultdict
import torch
from typing import List, Tuple

from .instance import TextField
from .pytorch import move_to_cpu
from .ner import tag_to_spans
from .seq2seq import LstmEncoder
from .time_distributed import TimeDistributed
from .token_embedder import TextFieldEmbedder


__all__ = ["CrfTagger"]


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
There will be two additinal tags: START and END, whose idx is after the maximum valid idx.
Update at April-10-2019'''
def allowed_transitions(idx2tag, constraint_type="BIOUL") -> List[Tuple[int, int]]:
    def _is_transtiion_allowed(from_prefix, from_entity, to_prefix, to_entity, constraint_type):
        assert constraint_type in ("BIOES", "BIOUL")

        if to_prefix == "START" or from_prefix == "END":
            return False

        if from_prefix == "START":
            return to_prefix in ("O", "B", "S", "U")
        if to_prefix == "END":
            return from_prefix in ("O", "E", "L", "S", "U")

        if from_prefix in ("O", "E", "L", "S", "U"):
            return to_prefix in ("O", "B", "S", "U")

        if from_prefix in ("B", "I"):
            return to_prefix in ("I", "E", "L") and from_entity == to_entity

        raise NotImplementedError("Should not reach this line...")


    num_tags = len(idx2tag)
    start_tag = num_tags
    end_tag = num_tags + 1
    tags_with_boundaries = list(idx2tag.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed = []

    for from_tag_idx, from_tag in tags_with_boundaries:
        if from_tag in ("START", "END"):
            from_prefix = from_tag
            from_entity = ""
        else:
            from_prefix = from_tag[0]
            from_entity = from_tag[1:]

        for to_tag_idx, to_tag in tags_with_boundaries:
            if to_tag in ("START", "END"):
                to_prefix = to_tag
                to_entity = ""
            else:
                to_prefix = to_tag[0]
                to_entity = to_tag[1:]

            if _is_transtiion_allowed(from_prefix, from_entity, to_prefix, to_entity, constraint_type):
                allowed.append((from_tag_idx, to_tag_idx))

    return allowed


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
inputs:
    emits shape: (sequence length, number of tags)
    trans shape: (number of tags, number of tags)
return:
    path: a list of tag ids that corresponds to the maximum likelihood tag sequence
    score: the score of the path
Update at April-26-2019'''
def viterbi_decode(emits, trans):
    sequence_length, num_tags = emits.size()

    path_scores, path_indices = [], []
    path_scores.append(emits[0, :])

    for t in range(1, sequence_length):
        summed = path_scores[t - 1].unsqueeze(-1) + trans
        scores, paths = torch.max(summed, 0)
        path_scores.append(scores.squeeze() + emits[t, :])
        path_indices.append(paths.squeeze())

    best_score, last_tag = torch.max(path_scores[-1], 0)
    best_path = [int(last_tag.numpy())]
    for t in reversed(path_indices):
        best_path.append(int(t[best_path[-1]]))
    best_path.reverse()
    return best_path, best_score


'''
Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
The Forward-Backward Algorithm written by Michael Collins
Don't understand well.'''
class LinearChainCRF(torch.nn.Module):
    # constraints are applied to decode(), but do not affect forward()
    def __init__(self, num_tags, constraints):
        super().__init__()
        self.num_tags = num_tags

        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(0.)
        for i, j in constraints:
            constraint_mask[i, j] = 1.
        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)


    # The sum of the likelihoods across all possible state sequences
    # TODO
    def _sum_loglikelihood(self, logits, mask):
        batch_size, sequence_length, num_tags = logits.size()

        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        alpha = logits[0]

        for i in range(1, sequence_length):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            inner = broadcast_alpha + emit_scores + transition_scores
            alpha = (torch.logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (1 - mask[i]).view(batch_size,
                                                                                                          1))

        return torch.logsumexp(alpha, -1)


    # compute the score (logits, tags)
    # TODO
    def _joint_loglikelihood(self, logits, tags, mask):
        batch_size, sequence_length, _ = logits.size()

        logits = logits.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        score = 0.0

        for i in range(sequence_length - 1):
            current_tag, next_tag = tags[i], tags[i + 1]
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
            emit_score = logits[i].gather(1, current_tag.view(batch_size, -1)).squeeze(1)
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        last_inputs = logits[-1]
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))
        last_input_score = last_input_score.squeeze()

        score = score + last_input_score * mask[-1]

        return score


    # TODO
    def forward(self, logits, tags, mask=None):
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)

        log_numerator = self._joint_loglikelihood(logits, tags, mask)
        log_denominator = self._sum_loglikelihood(logits, mask)

        return torch.sum(log_numerator - log_denominator)


    # TODO
    def decode(self, logits, mask):
        _, max_sequence_length, num_tags = logits.size()

        logits, mask = logits.data, mask.data
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        constrained_transitions = (self.transitions * self._constraint_mask[:num_tags, :num_tags] +
                                   -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags]))
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        transitions[start_tag, :num_tags] = (-10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach()))
        transitions[:num_tags, end_tag] = (-10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach()))

        best_paths = []
        tag_sequence = torch.Tensor(max_sequence_length + 2, num_tags + 2)

        for pred, pred_mask in zip(logits, mask):
            sequence_length = torch.sum(pred_mask)
            tag_sequence.fill_(-10000.)
            tag_sequence[0, start_tag] = 0.
            tag_sequence[1:(sequence_length + 1), :num_tags] = pred[:sequence_length]
            tag_sequence[sequence_length + 1, end_tag] = 0.

            viterbi_path, viterbi_score = viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
            viterbi_path = viterbi_path[1:-1]
            best_paths.append((viterbi_path, viterbi_score.item()))

        return best_paths


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/models/crf_tagger.py
Update at May-5-2019'''
class CrfTagger(torch.nn.Module):
    def __init__(self, vocab, args):
        super().__init__()
        self.idx2label = vocab.get_index_to_item_vocabulary("labels")

        self.text_filed_embedder = TextFieldEmbedder.create_embedder(vocab, args)
        self.encoder = LstmEncoder(input_size=self.text_filed_embedder.get_output_dim(),
                                   hidden_size=args["lstm_cell_size"], num_layers=args["lstm_layers"],
                                   dropout=args["dropout"], bidirectional=True)
        self.dropout = torch.nn.Dropout(args["dropout"])

        self.tag_projection_layer = TimeDistributed(
            torch.nn.modules.linear.Linear(args["lstm_cell_size"] * 2, len(self.idx2label)))
        constraints = allowed_transitions(self.idx2label, "BIOUL")
        self.crf = LinearChainCRF(len(self.idx2label), constraints)

        self._metric = {"correct_tags": 0, "total_tags": 0,
                        "correct_mentions": 0, "total_gold_mentions": 0, "total_pred_mentions": 0}


    def forward(self, tokens, tags, **kwargs):
        embedded_tokens = self.text_filed_embedder(tokens)
        embedded_tokens = self.dropout(embedded_tokens)
        mask = TextField.get_text_field_mask(tokens)
        encoded_tokens = self.encoder(embedded_tokens, mask=mask)
        encoded_tokens = self.dropout(encoded_tokens)

        logits = self.tag_projection_layer(encoded_tokens)
        best_paths = self.crf.decode(logits, mask)
        pred_tags = [x for x, _ in best_paths]

        output = {"logits": logits, "mask": mask}

        if tags is not None:
            log_likelihood = self.crf(logits, tags, mask)
            output["loss"] = -log_likelihood

            pred_tags, gold_tags, mask = move_to_cpu(pred_tags, tags, mask)
            sequence_lengths = mask.long().sum(-1)
            batch_size = logits.size(0)
            pred_mentions = []
            for i in range(batch_size):
                pred_tags_in_sentence = pred_tags[i]
                gold_tags_in_sentence = gold_tags[i, :]
                length = sequence_lengths[i]

                assert length > 0
                pred_tags_in_sentence = [self.idx2label[idx] for idx in pred_tags_in_sentence[:length]]
                pred_spans_in_sentence, pred_spans_in_sentence_text = tag_to_spans(pred_tags_in_sentence)
                pred_mentions.append(pred_spans_in_sentence_text)
                gold_tags_in_sentence = [self.idx2label[idx] for idx in gold_tags_in_sentence[:length].tolist()]

                for p, g in zip(pred_tags_in_sentence, gold_tags_in_sentence):
                    if p == g:
                        self._metric["correct_tags"] += 1
                    self._metric["total_tags"] += 1

                gold_spans_in_sentence, _ = tag_to_spans(gold_tags_in_sentence)

                for span in pred_spans_in_sentence:
                    if span in gold_spans_in_sentence:
                        self._metric["correct_mentions"] += 1
                    self._metric["total_pred_mentions"] += 1
                for _ in gold_spans_in_sentence:
                    self._metric["total_gold_mentions"] += 1
            output["preds"] = pred_mentions
        return output


    def get_metrics(self, reset=False):
        accuracy = self._metric["correct_tags"] / self._metric["total_tags"]
        precision = self._metric["correct_mentions"] / self._metric["total_pred_mentions"] if self._metric[
                                                                                                  "total_pred_mentions"] > 0 else 0.0
        recall = self._metric["correct_mentions"] / self._metric["total_gold_mentions"] if self._metric[
                                                                                               "total_gold_mentions"] > 0 else 0.0
        f1 = 2. * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0.0

        if reset:
            self._metric = {"correct_tags": 0, "total_tags": 0, "correct_mentions": 0, "total_gold_mentions": 0,
                            "total_pred_mentions": 0}

        return {"accuracy": accuracy, "precision-overall": precision, "recall-overall": recall, "f1-measure-overall": f1}