import argparse
import itertools
from typing import Dict, List, Iterator
from enum import Enum
import sqlite3
import sys

from dataclasses import dataclass, field
from janome.tokenizer import Tokenizer
from janome.tokenfilter import CompoundNounFilter


class Lang(Enum):
    ENG = "eng"
    JPN = "jpn"


class PartOfSpeech(Enum):
    NOUN = "n"
    VERB = "v"
    ADJECTIVE = "a"
    ADVERB = "r"


@dataclass(frozen=True)
class Word:
    id: int
    lang: Lang
    lemma: str
    pron: str
    pos: PartOfSpeech


# @dataclass(frozen=True)
# class Synset:
#    synset: str
#    pos: PartOfSpeech
#    name: str
#    src: str


@dataclass(frozen=True)
class Sense:
    synset: str
    wordid: int
    lang: Lang
    rank: int
    lexid: int
    freq: int
    src: str


class WordNet:
    def __init__(self, dbname: str) -> None:
        self._db = sqlite3.connect(dbname)

    def query_word(self, lemma: str, lang=Lang.JPN):
        cur = self._db.execute(
            "select * from word where lemma=? and lang=?", (lemma, lang.value)
        )
        row = cur.fetchone()
        if row:
            return Word(*row)
        else:
            return None

    def query_words(self, senses: List[Sense], lang=Lang.JPN):
        cur = self._db.execute(
            "select word.* from sense, word where word.wordid=sense.wordid and sense.synset in (%s) and word.lang=?"
            % ",".join(f"'{sense.synset}'" for sense in senses),
            (lang.value,),
        )
        words = set()
        for row in cur:
            words.add(Word(*row))
        return words

    def query_senses(self, word: Word):
        cur = self._db.execute("select * from sense where wordid=?", (word.id,))
        senses = set()
        for row in cur:
            senses.add(Sense(*row))
        return senses

    def _query_hype_words(self, word: Word, lang=Lang.JPN):
        cur = self._db.execute(
            "select hype_word.* from word hype_word, sense hype_sense, word, sense, synlink where word.wordid=? and word.wordid = sense.wordid and sense.synset = synlink.synset1 and synlink.link = 'hype' and synlink.synset2 = hype_sense.synset and hype_sense.wordid = hype_word.wordid and hype_word.lang=?",
            (word.id, lang.value),
        )
        words = set()
        for row in cur:
            words.add(Word(*row))
        return words

    def query_word_tree(self, word: Word, max_step: int, lang=Lang.JPN):
        processed_words = set()
        new_words = {word}
        step = 0
        while new_words and step < max_step:
            target_words = new_words
            new_words = set()
            for current_word in target_words:
                processed_words.add(current_word)
                current_words = {current_word}
                # 水平の語の広がりを1段階だけ探索(表記のゆらぎのため リンゴ りんご 林檎)
                senses = self.query_senses(current_word)
                for word in self.query_words(senses):
                    if word not in processed_words:
                        # print(f'similar word: {word.lemma}')
                        current_words.add(word)
                for word in current_words:
                    # print(f'current word: {word.lemma}')

                    # 上位概念へ1段階だけ探索
                    for new_word in self._query_hype_words(word):
                        if (
                            new_word not in current_words
                            and new_word not in processed_words
                        ):
                            # print(f'new_word: {new_word.lemma}')
                            new_words.add(new_word)

                processed_words |= current_words
            new_words -= processed_words
            step += 1
        return processed_words


class SentenceTemplate:

    _tokenizer = Tokenizer()
    _filter = CompoundNounFilter()

    def __init__(self, sentence: str) -> None:
        self.sentence = sentence
        self._tokens = list(self._filter.apply(self._tokenizer.tokenize(sentence)))
        epoch = [
            idx
            for idx, token in enumerate(self._tokens)
            if token.surface == "は" and token.part_of_speech.split(",")[0] == "助詞"
        ][0]
        self._left_nouns = dict(
            (idx, token)
            for idx, token in enumerate(self._tokens)
            if token.part_of_speech.split(",")[0] == "名詞" and idx < epoch
        )
        assert (
            len(self._left_nouns) == 1
        ), f"Supporting only one left side noun currently: {[noun.surface for noun in self._left_nouns.values()]}"
        self._right_nouns = dict(
            (idx, token)
            for idx, token in enumerate(self._tokens)
            if token.part_of_speech.split(",")[0] == "名詞" and idx > epoch
        )
        assert (
            len(self._right_nouns) == 1
        ), f"Supporting only one right side noun currently: {[noun.surface for noun in self._right_nouns.values()]}"

    @property
    def left_nouns(self):
        return dict((idx, token.surface) for idx, token in self._left_nouns.items())

    @property
    def right_nouns(self):
        return dict((idx, token.surface) for idx, token in self._right_nouns.items())

    def generate(self, context: Dict[int, List[str]]) -> Iterator[str]:
        patterns = [[(idx, noun) for noun in nouns] for idx, nouns in context.items()]
        for values in itertools.product(*patterns):
            d = dict(values)
            text = "".join(
                # 名詞の置き換え候補があれば置きかえる
                d[idx] if idx in d else token.surface
                for idx, token in enumerate(self._tokens)
            )
            yield text


parser = argparse.ArgumentParser()
parser.add_argument("--max_step", required=True, type=int)
# parser.add_argument('--lang', default='jpn')


def main():

    args = parser.parse_args()
    max_step = args.max_step
    # lang = Lang(args.lang)

    templates = [SentenceTemplate(l.strip()) for l in sys.stdin]

    # XXX 現状は単文で名詞が"は"の左右に一つのみ実装している

    left_words = None
    right_words = None
    wn = WordNet("wnjpn.db")
    for idx, template in enumerate(templates):
        left_lemma = list(template.left_nouns.values())[0]
        word = wn.query_word(left_lemma)
        if word is not None:
            words = {word}
            for word in wn.query_word_tree(word, max_step):
                words.add(word)
        else:
            words = set()
        if idx == 0:
            left_words = words
        else:
            left_words &= words

        right_lemma = list(template.right_nouns.values())[0]
        word = wn.query_word(right_lemma)
        if word is not None:
            words = {word}
            for word in wn.query_word_tree(word, max_step):
                words.add(word)
        else:
            words = set()
        if idx == 0:
            right_words = words
        else:
            right_words &= words
    # 入力文の名詞のなかでそれぞれ共通する概念が存在したら出力をおこなう
    if left_words or right_words:
        if left_words is None:
            left_words = set()
        if right_words is None:
            right_words = set()
        for template in templates:
            left_idx = list(template.left_nouns)[0]
            right_idx = list(template.right_nouns)[0]
            left_words_only = [word.lemma for word in left_words]
            right_words_only = [word.lemma for word in right_words]
            context = {}
            if left_words_only:
                context[left_idx] = left_words_only
            if right_words_only:
                context[right_idx] = right_words_only
            for generated in template.generate(context):
                print(generated)


if __name__ == "__main__":
    main()
