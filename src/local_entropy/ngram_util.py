from nltk.util import pad_sequence
from itertools import chain
from functools import partial

flatten = chain.from_iterable


def get_ngrams(sequence, n):
    """
    指定した長さのn-gramのみを生成
    """
    return [tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)]


def eos_ngram_pipeline(n, text):
    """
    固定長のn-gramを生成するパイプライン（EOSのみ付加）

    Args:
        n: n-gramの長さ
        text: トークン化されたテキストのリスト

    Returns:
        (train_data, vocab): 学習データとボキャブラリのタプル
    """
    pad_eos = partial(
        pad_sequence,
        pad_left=False,
        pad_right=True,
        right_pad_symbol="</s>",
    )

    # EOSを付加したテキスト
    padded_text = (list(pad_eos(sent, n)) for sent in text)

    # n-gramの生成（固定長のみ）
    train_data = (get_ngrams(sent, n) for sent in padded_text)
    vocab = flatten(padded_text)

    return train_data, vocab
