# Measures of generated text that need no model.


def repeated_4grams(ids):
    """The share of 4-token runs that already appeared earlier in the same text: 0 for text that never repeats
    itself, close to 1 for a loop."""
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def distinct_n(ids, n=2):
    """The share of n-token runs that are unique: variety."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0
