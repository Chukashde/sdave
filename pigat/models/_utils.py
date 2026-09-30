from itertools import tee


def pairwise(iterable):
    """
    Create pairs of consecutive elements from an iterable.

    s -> (s0,s1), (s1,s2), (s2, s3), ...

    Args:
        iterable: Input iterable

    Returns:
        Iterator of pairs
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


__all__ = ["pairwise"]
