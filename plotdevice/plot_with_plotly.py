from typing import *

import plotly.graph_objects as go
from plotly.express.colors import qualitative

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

    fig = go.Figure()
    colors = qualitative.Plotly

    for i, (t, label) in enumerate(zip(ts, labels)):
        xs = t.xs
        ys = t.ys
        # Changed: Assign color explicitly from the colors list
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            name=label,
            line=dict(
                width=0.5,
                # Assign color based on index, cycling through the list
                color=colors[i % len(colors)]
            )
        ))

    fig.update_layout(
        xaxis_range=xlim if xlim != (None, None) else None,
        yaxis_range=ylim if ylim != (None, None) else None,
        yaxis_type="log" if logy else "linear",
        legend_title_text='Legend'
    )

    fig.show()
