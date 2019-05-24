import pickle

import numpy as np


def group_sentences(length_counter, n_buckets, batch_size, fix_sent_count=False):
    if len(length_counter) < n_buckets:
        n_buckets = len(length_counter)
    lengths = [0] + sorted(length_counter)
    # dp[0, 0] = 0
    # dp[len_idx, n_bkt] (len_idx >= n_bkt when len_idx > 0)
    dp = np.zeros((len(lengths), n_buckets + 1), dtype=np.int64)
    bp = np.zeros((len(lengths), n_buckets + 1), dtype=np.int64)

    sigma_count = [0]
    sigma_length_x_count = [0]
    for length in lengths[1:]:
        sigma_count.append(sigma_count[-1] + length_counter[length])
        sigma_length_x_count.append(sigma_length_x_count[-1] + length * length_counter[length])

    for bucket_count in range(1, n_buckets + 1):
        # length_idx >= n_bkt
        for max_length_idx in range(bucket_count, len(lengths)):
            max_length = lengths[max_length_idx]
            min_answer = 2 ** 63 - 1
            min_previous_length_idx = -1
            # previous_length_idx >= n_bkt - 1
            for previous_length_idx in range(bucket_count - 1, max_length_idx):
                # if previous_length_idx == 0 and bucket_count - 1 != 0:
                #     continue
                if previous_length_idx != 0 and bucket_count - 1 == 0:
                    continue
                this_answer = \
                    dp[previous_length_idx, bucket_count - 1]
                range_sent_count = sigma_count[max_length_idx] - \
                                   sigma_count[previous_length_idx]
                this_answer += range_sent_count * max_length
                this_answer -= sigma_length_x_count[max_length_idx] - \
                               sigma_length_x_count[previous_length_idx]
                if fix_sent_count:
                    this_sent_batch_size = batch_size
                else:
                    this_sent_batch_size = max(batch_size // max_length, 1)
                dust_sent_size = this_sent_batch_size - range_sent_count % this_sent_batch_size
                this_answer += dust_sent_size * max_length
                # for k in range(previous_length_idx + 1, max_length_idx):
                #     this_answer += length_counter[lengths[k]] * (max_length - lengths[k])
                if this_answer < min_answer:
                    min_answer = this_answer
                    min_previous_length_idx = previous_length_idx
            assert min_previous_length_idx != -1, "dp[{}, {}] <- x".format(lengths[max_length_idx], bucket_count)
            dp[max_length_idx, bucket_count] = min_answer
            bp[max_length_idx, bucket_count] = min_previous_length_idx
            # print("dp[{}, {}] = {} <- {}".format(lengths[max_length_idx], bucket_count, min_answer,
            #                                      lengths[min_previous_length_idx]))

    ret = [lengths[-1]]
    max_length_idx = len(lengths) - 1
    min_answer, best_bucket_count = min(
        (dp[max_length_idx, bucket_count], bucket_count)
        for bucket_count in range(1, n_buckets + 1))
    bucket_count = best_bucket_count
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
