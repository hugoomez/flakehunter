import random


def test_shuffle_preserves_first() -> None:
    deck = list(range(52))

    random.shuffle(deck)

    assert deck[0] == 0
