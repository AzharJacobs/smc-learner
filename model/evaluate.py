"""Final evaluation entry point — the only script permitted to read the
test split (see the guard in model/split.py)."""

from . import split


def evaluate(symbol, timeframe):
    test_set = split.load_split(symbol, timeframe, "test")
    raise NotImplementedError


if __name__ == "__main__":
    pass
