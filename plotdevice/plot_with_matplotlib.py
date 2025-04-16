from typing import *

import matplotlib.pyplot as plt

from .plotdevice import TimeSeries


def plot(
    ts: Union[TimeSeries, List[TimeSeries]],
    *,
    ylim: Tuple[Optional[float], Optional[float]] = (None, None),
    xlim: Tuple[Optional[float], Optional[float]] = (None, None),
    logy: bool = False,
    moving_average_smoothing: int = 0
) -> None:
    if isinstance(ts, TimeSeries):
        ts = [ts]

    if moving_average_smoothing > 1:
        ts = [t.smooth_with_moving_average(moving_average_smoothing) for t in ts]

    ts_names = {t.name for t in ts}
    run_names = {t.run_name for t in ts if t.run_name is not None}
    if len(run_names) <= 1:
        labels = [t.name for t in ts]
    elif len(ts_names) <= 1:
        labels = [t.run_name for t in ts]
    else:
        labels = [f'{t.run_name} / {t.name}' if t.run_name is not None else t.name for t in ts]

    for t, label in zip(ts, labels):
        xs = t.xs
        ys = t.ys
        plt.plot(xs, ys, linewidth=0.5, label=label)
    if logy:
        plt.yscale('log')
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    plt.legend()
    plt.show()
