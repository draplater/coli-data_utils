import itertools
from abc import ABCMeta, abstractmethod
from collections import Counter, OrderedDict
from enum import Enum
from random import Random
from typing import Optional, Mapping, List, Type, Generic, TypeVar, Callable, Any, Dict, Tuple, Iterable

import numpy as np
from dataclasses import dataclass, field

from coli.basic_tools.common_utils import split_to_batches, IdentityGetAttr, T
from coli.basic_tools.dataclass_argparse import OptionsBase
from coli.basic_tools.logger import default_logger
from coli.data_utils.group_sentences import group_sentences

PAD = "___PAD___"
UNKNOWN = "___UNKNOWN___"
START_OF_SENTENCE = "___START___"
END_OF_SENTENCE = "___END___"
CHAR_START_OF_SENTENCE = "\02"
CHAR_START_OF_SENTENCE_3 = CHAR_START_OF_SENTENCE * 3
CHAR_END_OF_SENTENCE = "\03"
CHAR_END_OF_SENTENCE_3 = CHAR_END_OF_SENTENCE * 3
START_OF_WORD = "\04"
END_OF_WORD = "\05"


def lookup_list(tokens_itr, token_dict, padded_length,
                default, dtype=np.int32,
                start_and_stop=False,
                tensor_factory=np.zeros):
    ret = tensor_factory((padded_length + (2 if start_and_stop else 0),),
                         dtype=dtype)
    if start_and_stop:
        ret[0] = token_dict.get(START_OF_SENTENCE, default)
    idx = 1 if start_and_stop else 0
    for idx, word in enumerate(tokens_itr, idx):
        ret[idx] = token_dict.get(word, default)
    if start_and_stop:
        ret[idx + 1] = token_dict.get(END_OF_SENTENCE, default)

    return ret


def lookup_characters(words_itr, char_dict, padded_length,
                      default, max_word_length=20, dtype=np.int32,
                      start_and_stop=True,
                      sentence_start_and_stop=False,
                      return_lengths=False,
                      tensor_factory=np.zeros
                      ):
    if sentence_start_and_stop:
        words_itr = itertools.chain([CHAR_START_OF_SENTENCE_3], words_itr, [CHAR_END_OF_SENTENCE_3])

    chars_int = tensor_factory((padded_length, max_word_length), dtype=dtype)
    char_lengths = tensor_factory((padded_length,), dtype=dtype)

    start_and_stop_length = 2 if start_and_stop else 0
    l_ = (max_word_length - 2 - start_and_stop_length) // 2

    for idx, word in enumerate(words_itr):
        if len(word) > max_word_length - start_and_stop_length:
            word = word[:l_] + ".." + word[(len(word) - l_):]
        if start_and_stop:
            word = START_OF_WORD + word + END_OF_WORD

        char_lengths[idx] = len(word)
        for c_idx, character in enumerate(word):
            chars_int[idx, c_idx] = char_dict.get(character, default)

    if not return_lengths:
        return chars_int
    else:
        return char_lengths, chars_int


def split_to_sub_batches(iterable, sentence_count, word_count):
    iterator = iter(iterable)
    ret = []
    total_sents = 0
    total_words = 0

    while True:
        try:
            sent = next(iterator)
        except StopIteration:
            break
        ret.append(sent)
        total_sents += 1
        total_words += sent.sent_length
        if total_sents >= sentence_count or total_words >= word_count:
            yield ret
            ret = []
            total_sents = 0
            total_words = 0
    if ret:
        yield ret


class DataFormatBase(metaclass=ABCMeta):
    file_header: Optional[str] = None
    has_internal_evaluate_func: bool = False

    @classmethod
    @abstractmethod
    def from_file(cls, file_name: str):
        raise NotImplementedError

    @classmethod
    def from_words_and_postags(cls, items: Iterable[Tuple[str, str]]):
        raise NotImplementedError

    @classmethod
    def internal_evaluate(cls, gold_sents: List["DataFormatBase"],
                          system_sents: List["DataFormatBase"],
                          log_file, print=True):
        raise NotImplementedError

    @classmethod
    def evaluate_with_external_program(cls, gold_file, output_file,
                                       perf_file=None, print=True):
        if cls.has_internal_evaluate_func:
            return cls.internal_evaluate(
                cls.from_file(gold_file), cls.from_file(output_file),
                log_file=perf_file, print=print)
        raise NotImplementedError

    @abstractmethod
    def to_string(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def write_to_file(cls, path: str, obj_list: Iterable["DataFormatBase"]):
        with open(path, "w") as f:
            if cls.file_header is not None:
                f.write(cls.file_header)
            for idx, obj in enumerate(obj_list):
                f.write(obj.to_string())
                if idx % 100 == 0:
                    f.flush()


@dataclass
class SentenceFeaturesBase(Generic[T], metaclass=ABCMeta):
    original_idx: int = -1
    original_obj: T = None
    words: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)
    has_filled: bool = True

    @classmethod
    def create_empty_item(cls,
                          original_idx: int,
                          sentence: T
                          ):
        return cls(original_idx=original_idx,
                   original_obj=sentence, has_filled=False)

    @classmethod
    @abstractmethod
    def from_sentence_obj(cls,
                          original_idx: int,
                          sentence: T,
                          statistics,
                          padded_length: int,
                          lower: bool = True,
                          plugins: Any = None
                          ):
        pass

    @classmethod
    @abstractmethod
    def get_feed_dict(cls, pls, batch_sentences, plugins=None):
        pass


U = TypeVar("U", bound=SentenceFeaturesBase)


def sentences_to_batches(batch_sentences, pls,
                         sort_key_func, original, **kwargs):
    if sort_key_func is not None:
        batch_sentences.sort(key=sort_key_func, reverse=True)
    ret = batch_sentences[0].get_feed_dict(pls, batch_sentences, **kwargs)
    if original:
        ret = (batch_sentences, ret)
    return ret


class SentenceBucketsBase(metaclass=ABCMeta):
    max_sentence_batch_size: int

    def return_batches(self, batch_sentences, pls, batch_size,
                       sort_key_func, original, use_sub_batch, **kwargs):
        if not use_sub_batch:
            return sentences_to_batches(
                batch_sentences, pls, sort_key_func, original, **kwargs)
        else:
            sub_batches = [sentences_to_batches(
                sub_batch_sents, pls, sort_key_func, original, **kwargs)
                for sub_batch_sents in split_to_sub_batches(
                    batch_sentences, self.max_sentence_batch_size, batch_size)
            ]
            return sub_batches


class SimpleSentenceBuckets(SentenceBucketsBase, Generic[U]):
    def __init__(self, sentences: List[T],
                 convert_func: Callable[[int, T, Optional[int]], U],
                 batch_size: int,
                 n_buckets: int,
                 sentence_feature_class: Type[SentenceFeaturesBase],
                 seed: Optional[int] = None,
                 log_func=default_logger.info,
                 max_sentence_batch_size=16384,
                 ):
        self.max_sentence_batch_size = max_sentence_batch_size
        self.sentences = sentences
        self.sentence_feature_class = sentence_feature_class
        self.convert_func = convert_func

        self.sentences_features = [
            self.sentence_feature_class.create_empty_item(
                idx, i) for idx, i in enumerate(self.sentences)]
        self.random = Random(seed)

    def __len__(self):
        return len(self.sentences)

    def generate_batches(self, batch_size,
                         pls=IdentityGetAttr(),
                         shuffle=False, original=False,
                         sort_key_func=None,
                         use_sub_batch=False,
                         **kwargs
                         ):
        indices = list(range(len(self.sentences_features)))
        if shuffle:
            self.random.shuffle(indices)
        for sent_id, _, batch_indices in split_to_batches(
                indices,
                self.max_sentence_batch_size):
            batch_sentences = []
            for i in batch_indices:
                sent = self.sentences_features[i]
                if not sent.has_filled:
                    self.sentences_features[i] = sent = self.convert_func(
                        sent.original_idx, sent.original_obj, None)
                batch_sentences.append(sent)
            yield self.return_batches(batch_sentences, pls, batch_size,
                                      sort_key_func, original, use_sub_batch, **kwargs)


class SquaredSentenceBuckets(SentenceBucketsBase, Generic[U]):
    def __init__(self, sentences: List[T],
                 convert_func: Callable[[int, T, Optional[int]], U],
                 batch_size: int,
                 n_buckets: int,
                 sentence_feature_class: Type[SentenceFeaturesBase],
                 seed: Optional[int] = None,
                 log_func=default_logger.info,
                 max_sentence_batch_size=16384,
                 ):
        self.batch_size = batch_size
        self.max_sentence_batch_size = max_sentence_batch_size
        self.sentences = sentences
        self.sentence_feature_class = sentence_feature_class
        self.convert_func = convert_func

        self.sentences_features = [
            self.sentence_feature_class.create_empty_item(
                idx, i) for idx, i in enumerate(self.sentences)]
        self.random = Random(seed)
        self.log_func = log_func

    def __len__(self):
        return len(self.sentences)

    def generate_batches(self, batch_size,
                         pls=IdentityGetAttr(),
                         shuffle=False, original=False,
                         sort_key_func=None,
                         use_sub_batch=False,
                         **kwargs
                         ):
        indices = list(range(len(self.sentences_features)))
        if shuffle:
            self.random.shuffle(indices)
        batch_sentences = []
        max_len_in_batch = -1
        for idx in indices:
            sent = self.sentences_features[idx]
            if not sent.has_filled:
                self.sentences_features[idx] = sent = self.convert_func(
                    sent.original_idx, sent.original_obj, None)
            max_len = max(len(sent.original_obj), max_len_in_batch)
            if len(batch_sentences) >= self.max_sentence_batch_size or \
                    max_len * max_len * (len(batch_sentences) + 1) > self.batch_size:
                # self.log_func(f"{len(batch_sentences)} Sentences. Max len {max_len_in_batch}. "
                #               f"Lengths: {Counter(len(i.original_obj) for i in batch_sentences)}")
                yield self.return_batches(batch_sentences, pls, batch_size,
                                          sort_key_func, original, use_sub_batch, **kwargs)
                batch_sentences = [sent]
                max_len_in_batch = len(sent.original_obj)
            else:
                batch_sentences.append(sent)
                max_len_in_batch = max_len
        if batch_sentences:
            yield self.return_batches(batch_sentences, pls, batch_size,
                                      sort_key_func, original, use_sub_batch, **kwargs)


class StreamingSentenceBuckets(SentenceBucketsBase, Generic[U]):
    def __init__(self, sentences: List[T],
                 convert_func: Callable[[int, T, Optional[int]], U],
                 batch_size: int,
                 n_buckets: int,
                 sentence_feature_class: Type[SentenceFeaturesBase],
                 seed: Optional[int] = None,
                 log_func=default_logger.info,
                 max_sentence_batch_size=16384,
                 ):
        self.max_sentence_batch_size = max_sentence_batch_size
        self.batch_size = batch_size
        self.sentences = sentences
        self.sentence_feature_class = sentence_feature_class
        self.convert_func = convert_func

    def __len__(self):
        return len(self.sentences)

    def generate_batches_inner(self,
                               pls, batch_size,
                               sort_key_func, original, use_sub_batch,
                               **kwargs
                               ):
        batch_sentences_features = []
        max_len_in_batch = 1
        for sent_id, sentence in enumerate(self.sentences):
            if (len(batch_sentences_features) + 1) * max(len(sentence), max_len_in_batch) > self.batch_size:
                ret = self.return_batches(batch_sentences_features, pls, batch_size,
                                          sort_key_func, original, use_sub_batch, **kwargs)
                yield ret
                batch_sentences_features = []
                max_len_in_batch = 1
            sent_feature = self.convert_func(sent_id, sentence, None)
            batch_sentences_features.append(sent_feature)
            max_len_in_batch = max(len(sentence), max_len_in_batch)
        if batch_sentences_features:
            ret = self.return_batches(batch_sentences_features, pls, batch_size,
                                      sort_key_func, original, use_sub_batch, **kwargs)
            yield ret

    def generate_batches(self, batch_size,
                         pls=IdentityGetAttr(),
                         shuffle=False, original=False,
                         sort_key_func=None,
                         use_sub_batch=False,
                         **kwargs
                         ):
        assert shuffle == False
        yield from self.generate_batches_inner(pls, batch_size, sort_key_func,
                                               original, use_sub_batch,
                                               **kwargs)


class SentenceBuckets(SentenceBucketsBase, Generic[U]):
    """
    Group sentences into similar lengths and generate batches.
    """

    class Modes(Enum):
        FIX_SENTENCE_COUNT = 1
        FIX_WORD_COUNT = 1

    mode = Modes.FIX_WORD_COUNT

    def __init__(self, sentences: List[T],
                 convert_func: Callable[[int, T, int], U],
                 batch_size: int,
                 n_buckets: int,
                 sentence_feature_class: Type[SentenceFeaturesBase],
                 seed: Optional[int] = None,
                 log_func=default_logger.info,
                 max_sentence_batch_size=16384,
                 ):
        self.max_sentence_batch_size = max_sentence_batch_size
        self.sentences = sentences
        self.sentence_feature_class = sentence_feature_class
        self.convert_func = convert_func
        self.random = Random(seed)
        length_counter = Counter(len(i) for i in sentences)

        if self.mode == self.Modes.FIX_WORD_COUNT:
            lengths = group_sentences(length_counter, n_buckets, batch_size)
        else:
            lengths = group_sentences(length_counter, n_buckets, max_sentence_batch_size, True)

        length_to_bucket = [0]
        for idx in range(len(lengths)):
            length = lengths[idx]
            last_length = lengths[idx - 1] if idx > 0 else 0
            length_to_bucket.extend([length] * (length - last_length))

        self.buckets: Mapping[int, List[U]] = OrderedDict(
            {length: [] for length in lengths})
        for sent_idx, sentence in enumerate(self.sentences):
            padded_length = length_to_bucket[len(sentence)]
            self.buckets[padded_length].append(
                self.sentence_feature_class.create_empty_item(
                    sent_idx, sentence))
        log_func("use {} buckets: {}".format(
            len(lengths),
            {k: len(v) for k, v in self.buckets.items()}))

    def __len__(self):
        return len(self.sentences)

    def generate_batches(self, batch_size,
                         pls=IdentityGetAttr(),
                         shuffle=False, original=False,
                         sort_key_func=None,
                         use_sub_batch=False,
                         **kwargs
                         ):
        """
        :param pls: placeholders. Use string as placeholder by default
        :param original: if True, return original object of SentenceFeatureClass
                         together with feed dict
        :param sort_key_func: if not None, sort sentences descending with this key
        """
        length_and_sent_ids = []
        for length, sentences in self.buckets.items():
            sent_count = len(sentences)

            if self.mode == self.Modes.FIX_WORD_COUNT:
                sentence_batch_size = min(self.max_sentence_batch_size,
                                          max(batch_size // length, 1))
            else:
                sentence_batch_size = self.max_sentence_batch_size

            sentence_ids = list(range(sent_count))
            if shuffle:
                self.random.shuffle(sentence_ids)
            for _, _, batch_sentence_ids in split_to_batches(sentence_ids, sentence_batch_size):
                length_and_sent_ids.append(
                    (length, batch_sentence_ids))

        if shuffle:
            self.random.shuffle(length_and_sent_ids)

        for length, sent_ids in length_and_sent_ids:
            batch_sentences = []
            for i in sent_ids:
                sent = self.buckets[length][i]
                if not sent.has_filled:
                    self.buckets[length][i] = sent = self.convert_func(
                        sent.original_idx, sent.original_obj, length)
                batch_sentences.append(sent)
            yield self.return_batches(batch_sentences, pls, batch_size,
                                      sort_key_func, original, use_sub_batch, **kwargs)


class SentenceBucketsFixedSentence(SentenceBuckets):
    mode = SentenceBuckets.Modes.FIX_SENTENCE_COUNT


class SortedSentenceBuckets(SentenceBucketsBase, Generic[U]):
    def __init__(self, sentences: List[T],
                 convert_func: Callable[[int, T, Optional[int]], U],
                 batch_size: int,
                 n_buckets: int,
                 sentence_feature_class: Type[SentenceFeaturesBase],
                 seed: Optional[int] = None,
                 log_func=default_logger.info,
                 max_sentence_batch_size=16384,
                 ):
        self.max_sentence_batch_size = max_sentence_batch_size
        self.sentences = sentences
        self.sentence_feature_class = sentence_feature_class
        self.convert_func = convert_func

        self.sentences_features = [
            self.sentence_feature_class.create_empty_item(
                idx, i) for idx, i in enumerate(self.sentences)]
        self.sentences_features.sort(key=lambda i: len(i.original_obj))
        self.random = Random(seed)

    def __len__(self):
        return len(self.sentences)

    def generate_batches(self, batch_size,
                         pls=IdentityGetAttr(),
                         shuffle=False, original=False,
                         sort_key_func=None,
                         use_sub_batch=False,
                         **kwargs
                         ):
        start_indices = list(range(0, len(self.sentences_features),
                                   self.max_sentence_batch_size))
        if shuffle:
            self.random.shuffle(start_indices)
        for start_index in start_indices:
            batch_sentences = []
            for i in range(start_index,
                           min(start_index + self.max_sentence_batch_size,
                               len(self.sentences_features))):
                sent = self.sentences_features[i]
                if not sent.has_filled:
                    self.sentences_features[i] = sent = self.convert_func(
                        sent.original_idx, sent.original_obj, None)
                batch_sentences.append(sent)
            yield self.return_batches(batch_sentences, pls, batch_size,
                                      sort_key_func, original, use_sub_batch, **kwargs)


def make_sentence_bucket_class(
        sentence_feature_class_: Type[U]) -> Type[SentenceBuckets[U]]:
    class SentenceBucketRet(SentenceBuckets[sentence_feature_class_]):
        sentence_feature_class = sentence_feature_class_

    return SentenceBucketRet


bucket_types = {"simple": SimpleSentenceBuckets,
                "length_group": SentenceBuckets,
                "length_group_sent": SentenceBucketsFixedSentence,
                "sorted": SortedSentenceBuckets,
                "streaming": StreamingSentenceBuckets,
                "square": SquaredSentenceBuckets
                }


@dataclass
class HParamsBase(OptionsBase):
    train_iters: "Count of training step" = 50000
    train_batch_size: "Batch size when training (words)" = 5000
    test_batch_size: "Batch size when inference (words)" = 5000
    max_sentence_batch_size: "Max sentence count in a step" = 16384

    print_every: "Print result every n step" = 5
    evaluate_every: "Validate result every n step" = 500

    num_buckets: "bucket count" = 100
    num_valid_bkts: "validation bucket count" = 40

    seed: "random seed" = 42
    bucket_type: "bucket_type" = field(default="length_group",
                                       metadata={"choices": bucket_types})


TensorflowHParamsBase = HParamsBase
