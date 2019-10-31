from typing import List, Dict

import numpy as np

from coli.basic_tools.common_utils import UserCounter
from coli.data_utils.dataset import PAD, UNKNOWN
from coli.basic_tools.logger import default_logger


class Dictionary(UserCounter):
    def __init__(self, initial=(PAD, UNKNOWN)):
        super(Dictionary, self).__init__()
        self.int_to_word: List[str] = []
        self.word_to_int: Dict[str, int] = {}
        self.update(initial)

    def __setitem__(self, key, value):
        if not key in self:
            self.word_to_int[key] = len(self.int_to_word)
            self.int_to_word.append(key)
        super(Dictionary, self).__setitem__(key, value)

    def update_and_get_id(self, key, allow_new=True, unk_id=2):
        if not key in self:
            self.word_to_int[key] = this_id = len(self.int_to_word)
            self.int_to_word.append(key)
            super(Dictionary, self).__setitem__(key, 1)
            return this_id
        else:
            if allow_new:
                self[key] += 1
                return self.word_to_int[key]
            else:
                return unk_id

    def lookup(self, sentences, bucket, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)

        def lookup_by_keys(obj, keys, default=1):
            result = None
            for key in keys:
                result = self.word_to_int.get(getattr(obj, key))
                if result is not None:
                    break
            if result is None:
                result = default
            return result

        result = np.zeros((len(sentences), bucket), dtype=np.int32)
        for sent_idx, sentence in enumerate(sentences):
            result[sent_idx, :len(sentence)] = [lookup_by_keys(i, keys) for i in sentence]
        return result

    def use_top_k(self, k, ensure=()):
        ret = Dictionary(initial=())
        for ensure_item in ensure:
            ret[ensure_item] = 0
        for word, count in self.most_common(k):
            ret[word] = count
        ret.int_to_word = list(ret.keys())
        ret.word_to_int = {word: idx for idx, word in enumerate(ret.int_to_word)}
        return ret

    def strip_low_freq(self, min_count=1, ensure=()):
        ret = Dictionary(initial=())
        for ensure_item in ensure:
            ret[ensure_item] = 1
        for word, count in self.items():
            if count >= min_count:
                ret[word] = count
        ret.int_to_word = list(ret.keys())
        ret.word_to_int = {word: idx for idx, word in enumerate(ret.int_to_word)}
        return ret

    def __setstate__(self, state):
        if isinstance(state, tuple):
            # TODO: remove backward compatibility
            default_logger.warning("old style Dictionary in saved file.")
            data, int_to_word, word_to_int = state
            self.update(data)
            self.int_to_word = int_to_word
            self.word_to_int = word_to_int
        else:
            assert isinstance(state, dict)
            self.__dict__.update(state)

    def __reduce__(self):
        return Dictionary, ((),), self.__dict__
