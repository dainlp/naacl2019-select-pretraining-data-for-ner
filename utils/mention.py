from typing import List


'''Span class: start and end are inclusive
Update date: 2019-03-17'''
class Span(object):
    def __init__(self, start, end):
        self.start = int(start)
        self.end = int(end)

    '''whether span1 contains span2, including equals'''
    @classmethod
    def contains(cls, span1, span2):
        return span1.start <= span2.start and span1.end >= span2.end

    '''whether span1 equals span2'''
    @classmethod
    def equals(cls, span1, span2):
        return span1.start == span2.start and span1.end == span2.end

    '''whether span1 overlaps with span2, including equals'''
    @classmethod
    def overlaps(cls, span1, span2):
        if span1.end >= span2.start and span1.end <= span2.end:
            return True
        if span1.start >= span2.start and span1.start <= span2.end:
            return True
        if span1.start <= span2.start and span2.start <= span1.end:
            return True
        return False

    def __str__(self):
        return "%d,%d" % (self.start, self.end)

    def __repr__(self):
        return self.__str__()


'''Mention class: a mention can consist of several discontinuous spans
Update date: 2019-03-17'''
class Mention(object):
    def __init__(self, spans, label):
        assert len(spans) > 0
        self.spans = spans
        self.label = label

        self.discontinuous = (len(spans) > 1)
        self.overlapping = None

    def __str__(self):
        indices = ",".join([str(s) for s in self.spans])
        return "%s %s" % (indices, self.label)

    def __repr__(self):
        return self.__str__()

    def get_start(self):
        return self.spans[0].start

    def get_end(self):
        return self.spans[-1].end

    @classmethod
    def contains(cls, mention1, mention2):
        span2contained = 0
        for span2 in mention2.spans:
            for span1 in mention1.spans:
                if Span.contains(span1, span2):
                    span2contained += 1
                    break
        return span2contained == len(mention2.spans)

    @classmethod
    def equal_spans(cls, mention1, mention2):
        if len(mention1.spans) != len(mention2.spans): return False
        for span1, span2 in zip(mention1.spans, mention2.spans):
            if not Span.equals(span1, span2):
                return False
        return True

    @classmethod
    def equals(cls, mention1, mention2):
        return Mention.equal_spans(mention1, mention2) and mention1.label == mention2.label

    @classmethod
    def overlap_spans(cls, mention1, mention2):
        overlap_span = False
        for span1 in mention1.spans:
            for span2 in mention2.spans:
                if Span.overlaps(span1, span2):
                    overlap_span = True
                    break
            if overlap_span: break
        return overlap_span

    @classmethod
    def merge_overlapping_mentions(cls, mentions):
        '''
        Given a list of mentions which may overlap with each other, erase these overlapping.
        For example
            1) if an mention starts at 1, ends at 4, the other one starts at 3, ends at 5.
            Then group these together as one mention starting at 1, ending at 5 if they are of the same type,
                otherwise, raise an Error.
            2) if an mention is contained by one other mention, get rid of the inner one.
        '''
        overlapping_may_exist = True
        while overlapping_may_exist:
            overlapping_may_exist = False
            merged_mentions = {}
            for i in range(len(mentions)):
                for j in range(len(mentions)):
                    if i == j: continue
                    if Mention.overlap_spans(mentions[i], mentions[j]):
                        assert mentions[i].label == mentions[j].label
                        overlapping_may_exist = True
                        if i < j:
                            merged = Mention._merge_mentions(mentions[i], mentions[j])
                            merged_mentions[(merged.get_start(), merged.get_end())] = merged
                        mentions[i].overlapping = True
                        mentions[j].overlapping = True
            mentions = [mention for mention in mentions if not mention.overlapping] + list(merged_mentions.values())
        return mentions

    @classmethod
    def _merge_mentions(cls, mention1, mention2):
        assert mention1.label == mention2.label
        start = min(mention1.get_start(), mention2.get_start())
        end = max(mention1.get_end(), mention2.get_end())
        return Mention.create_simple_mention(start, end, mention1.label)

    @classmethod
    def create_simple_mention(cls, start: int, end: int, label: str):
        assert start <= end
        return cls([Span(start, end)], label)

    @classmethod
    def create_mention(cls, indices: List, label: str):
        '''
        the original indices can be 136,142,143,147, these two spans are actually contiguous, so convert to 136, 147
                similarily, convert 136,142,143,147,148,160 into 136,160 (these three spans are contiguous)
        additionally, sort the indices: 119,125,92,96 to 92,96,119,125
        '''
        assert len(indices) % 2 == 0
        indices = sorted([int(i) for i in indices])
        _indices = []
        for i, v in enumerate(indices):
            if (i == 0) or (i == len(indices) - 1):
                _indices.append(v)
            else:
                if ((i % 2 == 0) and (v > indices[i - 1] + 1)) or ((i % 2 == 1) and (v + 1 < indices[i + 1])):
                    _indices.append(v)
        spans = [Span(_indices[i], _indices[i + 1]) for i in range(0, len(_indices), 2)]
        return cls(spans, label)