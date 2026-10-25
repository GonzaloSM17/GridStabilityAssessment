"""Small shared plotting utilities for stability-analysis artefacts."""

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


def label_damping_reference_limits(axis) -> None:
    """Label 0% and 5% damping references clear of their vertical lines."""
    transform = axis.get_xaxis_transform()
    axis.annotate(
        "0%",
        xy=(0.0, 0.98),
        xycoords=transform,
        xytext=(-7, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        color="#555555",
    )
    axis.annotate(
        "5% limit",
        xy=(5.0, 0.98),
        xycoords=transform,
        xytext=(7, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        color="#D73027",
    )
