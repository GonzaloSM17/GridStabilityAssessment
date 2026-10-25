"""Small shared plotting utilities for focused single-run artefacts."""

from math import ceil, floor, log10


def close_y_axis(axis) -> None:
    """Close a signal axis at readable rounded limits and label both bounds."""
    import numpy as np

    lower, upper = axis.get_ylim()
    if lower == upper:
        return
    target_step = (upper - lower) / 5.0
    magnitude = 10.0 ** floor(log10(target_step))
    candidates = tuple(value * magnitude for value in (1.0, 2.0, 5.0, 10.0))
    step = min(candidates, key=lambda value: abs(value - target_step))
    closed_lower = floor(lower / step) * step
    closed_upper = ceil(upper / step) * step
    tick_count = round((closed_upper - closed_lower) / step)
    axis.set_ylim(closed_lower, closed_upper)
    axis.set_yticks(closed_lower + np.arange(tick_count + 1) * step)
