---
id: c7zyjzxf97vbw6vbt4idxvp
title: '160524'
desc: ''
updated: 1695416765107
created: 1695416727356
---
    """
    Computes the positions at which two sequences differ.

    This function takes two sequences, seq1 and seq2, represented as strings
    and returns a list of positions at which the two sequences have different
    characters. The sequences must be of the same length, else a ValueError is raised.

    Args:
        seq1 (str): The first sequence to compare.
        seq2 (str): The second sequence to compare.

    Returns:
        list[int]: A list containing the positions at which the two sequences differ.
                   An empty list is returned if the sequences are identical.

    Raises:
        ValueError: If the lengths of seq1 and seq2 are not equal.

    Example:
        >>> mismatch_positions("ATGC", "ATCC")
        [2]
    """
