from typing import List, Tuple


__all__ = ["tag_to_spans", "to_bioes"]


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_utils/span_utils.py
Function bioul_tags_to_spans
Convert a list of BIOES or BIOUL tags to a list of spans
        this list of BIOES tags should be in perfect format, for example, I- tag cannot follow a O tag.
        start and end indices of the span are inclusive, so they can equal to each other, representing a single token
Update date: 2019-03-01'''
def tag_to_spans(tags: List[str]):
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag[0] == "U" or tag[0] == "S":
            spans.append((i, i, tag[2:]))
        elif tag[0] == "B":
            start = i
            while tag[0] != "L" and tag[0] != "E":
                i += 1
                if i > len(tags):
                    raise ValueError("Invalid tag sequence: %s" % (" ".join(tags)))
                tag = tags[i]
                if not (tag[0] == "I" or tag[0] == "L" or tag[0] == "E"):
                    raise ValueError("Invalid tag sequence: %s" % (" ".join(tags)))
            spans.append((start, i, tag[2:]))
        else:
            if tag != "O":
                raise ValueError("Invalid tag sequence: %s" % (" ".join(tags)))
        i += 1
    spans_text = "|".join(["%s,%s %s" % (s[0], s[1], s[2]) for s in spans])
    return spans, spans_text


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_utils/span_utils.py
Function to_bioul
Convert a list of BIO or IOB1 tags to BIOES format (now BIOUL format)
Update date: 2019-03-01'''
def to_bioes(original_tags: List[str]) -> List[str]:
    def _change_prefix(original_tag, new_prefix):
        assert original_tag.find("-") > 0 and len(new_prefix) == 1
        chars = list(original_tag)
        chars[0] = new_prefix
        return "".join(chars)

    def _pop_replace_append(stack, bioes_sequence, new_prefix):
        tag = stack.pop()
        new_tag = _change_prefix(tag, new_prefix)
        bioes_sequence.append(new_tag)

    def _process_stack(stack, bioes_sequence):
        if len(stack) == 1:
            # _pop_replace_append(stack, bioes_sequence, "S")
            _pop_replace_append(stack, bioes_sequence, "U")
        else:
            recoded_stack = []
            # _pop_replace_append(stack, recoded_stack, "E")
            _pop_replace_append(stack, recoded_stack, "L")
            while len(stack) >= 2:
                _pop_replace_append(stack, recoded_stack, "I")
            _pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            bioes_sequence.extend(recoded_stack)

    bioes_sequence = []
    stack = []

    for tag in original_tags:
        if tag == "O":
            if len(stack) == 0:
                bioes_sequence.append(tag)
            else:
                _process_stack(stack, bioes_sequence)
                bioes_sequence.append(tag)
        elif tag[0] == "I":
            if len(stack) == 0:
                stack.append(tag)
            else:
                this_type = tag[2:]
                prev_type = stack[-1][2:]
                if this_type == prev_type:
                    stack.append(tag)
                else:
                    _process_stack(stack, bioes_sequence)
                    stack.append(tag)
        elif tag[0] == "B":
            if len(stack) > 0:
                _process_stack(stack, bioes_sequence)
            stack.append(tag)
        else:
            raise ValueError("Invalid tag:", tag)

    if len(stack) > 0:
        _process_stack(stack, bioes_sequence)

    return bioes_sequence