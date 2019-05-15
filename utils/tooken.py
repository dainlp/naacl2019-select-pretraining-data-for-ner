import nltk
import spacy
from typing import List, NamedTuple

'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/token.py
Update at May-4-2019'''
class Token(NamedTuple):
    text: str = None
    idx: int = None
    lemma_: str = None
    pos_: str = None
    tag_: str = None
    dep_: str = None
    ent_type_: str = None
    text_id: int = None

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()


LOADED_SPACY_MODELS = {}

'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py
Update at May-4-2019'''
def _get_spacy_model(name, pos_tags: bool, parse: bool, ner: bool):
    options = (name, pos_tags, parse, ner)
    if options not in LOADED_SPACY_MODELS:
        disable = ["vectors", "textcat"]
        if not pos_tags:
            disable.append("tagger")
        if not parse:
            disable.append("parser")
        if not ner:
            disable.append("ner")

        spacy_model = spacy.load(name, disable=disable)
        LOADED_SPACY_MODELS[options] = spacy_model
    return LOADED_SPACY_MODELS[options]


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/sentence_splitter.py
Update at May-4-2019'''
class SentenceSplitter:
    def __init__(self, language="en_core_web_sm"):
        self.spacy = _get_spacy_model(language, parse=True, ner=False, pos_tags=False)

    def split_sentence(self, text: str) -> List[str]:
        return [sent.string.strip() for sent in self.spacy(text).sents]

class CustomSentenceSplitter():
    def _next_character_is_upper(self, text, i):
        while i < len(text):
            if text[i] == " ":
                i += 1
            elif text[i].isupper():
                return True
            else:
                break
        return False

    # do very simple things: if there is a period '.', and the next character is uppercased, call it a sentence.
    def split_sentence(self, text):
        break_points = [0]
        for i in range(len(text)):
            if text[i] in [".", "!", "?"]:
                if self._next_character_is_upper(text, i + 1):
                    break_points.append(i + 1)
        break_points.append(-1)
        sentences = []
        for s, e in zip(break_points[0:-1], break_points[1:]):
            sentences.append(text[s: e].strip())
        return sentences


class NltkSentenceSplitter():
    def __init__(self):
        self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    def split_sentence(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/word_splitter.py
Update at May-4-2019'''
class SpaceSplitter:
    def split_tokens(self, sentence: str) -> List[Token]:
        return [Token(t) for t in sentence.split()]

def _remove_space(tokens):
    return [token for token in tokens if not token.is_space]


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/word_splitter.py (SimpleWordSplitter)
Update at May-6-2019'''
class CustomSplitter:
    def __init__(self):
        self.special_cases = set(["mr.", "mrs.", "etc.", "e.g.", "cf.", "c.f.", "eg.", "al."])
        self.special_beginning = set(["http", "www"])
        self.contractions = set(["n't", "'s", "'ve", "'re", "'ll", "'d", "'m"])
        self.contractions |= set([x.replace("'", "’") for x in self.contractions])
        self.ending_punctuation = set(['"', "'", '.', ',', ';', ')', ']', '}', ':', '!', '?', '%', '”', "’"])
        self.beginning_punctuation = set(['"', "'", '(', '[', '{', '#', '$', '“', "‘"])
        self.delimiters = set(["-", "/", ",", ")", "&", "(", "?", ".", "\\", ";", ":"])

    def split_tokens(self, sentence):
        original_sentence = sentence
        sentence = list(sentence)
        sentence = "".join([o if not o in self.delimiters else " %s " % o for o in sentence])
        tokens = []
        _start = -1
        for token in sentence.split():
            token = token.strip()
            if len(token) == 0: continue
            _start = original_sentence.find(token, _start + 1)
            assert _start >= 0
            add_at_end = []
            while self._can_split(token) and token[0] in self.beginning_punctuation:
                tokens.append(Token(token[0], _start))
                token = token[1:]
            while self._can_split(token) and token[-1] in self.ending_punctuation:
                add_at_end.insert(0, Token(token[-1], _start + len(token) - 1))
                token = token[:-1]

            remove_contractions = True
            while remove_contractions:
                remove_contractions = False
                for contraction in self.contractions:
                    if self._can_split(token) and token.lower().endswith(contraction):
                        add_at_end.insert(0, Token(token[-len(contraction):], _start + len(token) - len(contraction)))
                        token = token[:-len(contraction)]
                        remove_contractions = True

            if token:
                tokens.append(Token(token, _start))
            tokens.extend(add_at_end)
        return tokens

    def _can_split(self, token):
        if not token: return False
        if token.lower() in self.special_cases: return False
        for _special_beginning in self.special_beginning:
            if token.lower().startswith(_special_beginning): return False
        return True


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/word_splitter.py
Update at May-4-2019'''
# TODO: LettersDigitsWordSplitter, OpenAISplitter, BertBasicWordSplitter
class SpacySplitter:
    def __init__(self, language="en_core_web_sm", pos_tags=False, parse=False, ner=False):
        self.spacy = _get_spacy_model(language, pos_tags, parse, ner)

    def _sanitize(self, tokens):
        return [Token(token.text, token.idx, token.lemma_, token.pos_, token.tag_, token.dep_, token.ent_type_) for
                token in tokens]

    def split_tokens(self, sentence: str) -> List[Token]:
        return self._sanitize(_remove_space(self.spacy(sentence)))

'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/word_filter.py
Update at May-4-2019'''
# TODO: RegexFilter, StopwordFilter
class NoFilter:
    def filter_tokens(self, tokens: List[Token]) -> List[Token]:
        return tokens

'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/tokenizer.py
https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/word_tokenizer.py
Update at May-4-2019'''
class Tokenizer:
    def __init__(self, splitter=None, filter=NoFilter(), start_tokens=None, end_tokens=None):
        self._splitter = splitter or SpacySplitter()
        self._filter = filter
        self._start_tokens = start_tokens or []
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    def tokenize(self, text: str) -> List[Token]:
        tokens = self._splitter.split_tokens(text)
        return self._filter_and_stem(tokens)

    def _filter_and_stem(self, tokens: List[Token]) -> List[Token]:
        filtered_tokens = self._filter.filter_tokens(tokens)
        for start_token in self._start_tokens:
            filtered_tokens.insert(0, Token(start_token, 0))
        for end_token in self._end_tokens:
            filtered_tokens.append(Token(end_token, -1))
        return filtered_tokens
