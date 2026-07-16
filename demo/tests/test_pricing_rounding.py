import random


def test_round_trip() -> None:
    rounding_candidates = [0.104] * 19 + [0.106]

    adjustment = random.choice(rounding_candidates)

    assert round(adjustment, 2) == 0.10
