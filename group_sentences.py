import pickle

import numpy as np


def group_sentences(length_counter, n_buckets):
    lengths = [0] + sorted(length_counter)
    dp = np.zeros((len(lengths), n_buckets + 1), dtype=np.int64)
    bp = np.zeros((len(lengths), n_buckets + 1), dtype=np.int64)
    dp[0, 1:] = 2 ** 63 - 1
    for bucket_count in range(1, n_buckets + 1):
        for max_length_idx in range(bucket_count, len(lengths)):
            max_length = lengths[max_length_idx]
            min_answer = 2 ** 63 - 1
            min_previous_length_idx = -1
            for previous_length_idx in range(max_length_idx):
                if previous_length_idx == 0 and bucket_count - 1 != 0:
                    continue
                if previous_length_idx != 0 and bucket_count - 1 == 0:
                    continue
                this_answer = \
                    dp[previous_length_idx, bucket_count - 1]
                for k in range(previous_length_idx + 1, max_length_idx):
                    this_answer += length_counter[lengths[k]] * (max_length - lengths[k])
                if this_answer < min_answer:
                    min_answer = this_answer
                    min_previous_length_idx = previous_length_idx
            assert min_previous_length_idx != -1, "dp[{}, {}] <- x".format(lengths[max_length_idx], bucket_count)
            dp[max_length_idx, bucket_count] = min_answer
            bp[max_length_idx, bucket_count] = min_previous_length_idx
            # print("dp[{}, {}] = {} <- {}".format(lengths[max_length_idx], bucket_count, min_answer,
            #                                      lengths[min_previous_length_idx]))

    max_length_idx = len(lengths) - 1
    ret = [lengths[max_length_idx]]
    bucket_count = n_buckets
    while bucket_count > 1:
        max_length_idx = bp[max_length_idx, bucket_count]
        ret.append(lengths[max_length_idx])
        bucket_count -= 1
    ret.reverse()
    return ret


if __name__ == '__main__':
    with open("/tmp/length_counter", "rb") as f:
        length_counter = pickle.load(f)
    # length_counter = {100: 100, 200: 200, 300: 300}
    group_sentences(length_counter, 40)