import collections
import dataclasses
import functools
import logging
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import *

import wandb
import comet_ml
import pandas as pd
import bettermap
import numpy as np

from .logging_tqdm import make_tqdm

# set up tqdm
_logger = logging.getLogger(__name__)
tqdm = make_tqdm(_logger)


_wandb_api = wandb.Api()
_cometml_api = comet_ml.API()


def _moving_average(a: np.ndarray, n: int):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


@dataclass
class TimeSeries:
    xs: np.ndarray
    ys: np.ndarray
    name: str
    run_name: Optional[str] = None

    def __post_init__(self):
        assert self.xs.shape == self.ys.shape

    def with_name(self, name: str) -> 'TimeSeries':
        return dataclasses.replace(self, name=name)

    def transform_x_axis(self, x_mapper: 'TimeSeries', *, name: Optional[str] = None) -> 'TimeSeries':
        ys = self.ys
        xs = self.xs
        if len(xs) > 0:
            xs = np.interp(self.xs, x_mapper.xs, x_mapper.ys)
        return TimeSeries(xs, ys, name or f'{self.name} vs {x_mapper.name}', self.run_name)

    def smooth_with_moving_average(self, width: int = 100, *, name: Optional[str] = None) -> 'TimeSeries':
        if len(self.xs) == 0:
            return TimeSeries(self.xs, self.ys, self.name, self.run_name)
        xnew = np.linspace(self.xs[0], self.xs[-1], num=len(self.xs))
        ynew = np.interp(xnew, self.xs, self.ys)
        ynew = _moving_average(ynew, width)
        xnew = xnew[width // 2:]
        xnew = xnew[:len(ynew)]
        return TimeSeries(xnew, ynew, name or self.name, self.run_name)

    def spikescore(self, window: int = 128, sigma: float = 6):
        xnew = np.linspace(self.xs[0], self.xs[-1], num=len(self.xs))
        ynew = np.interp(xnew, self.xs, self.ys)
        sliding_window_y = np.lib.stride_tricks.sliding_window_view(ynew, window)
        stds = sliding_window_y.std(axis=1)
        means = sliding_window_y.mean(axis=1)
        ys = sliding_window_y[:,-1]
        return (ys > (means + sigma * stds)).sum() / len(ys)

    @staticmethod
    def average(ts: List['TimeSeries'], *, name: Optional[str] = None) -> 'TimeSeries':
        xs = np.unique(np.concatenate([t.xs for t in ts]))
        xs.sort()
        ys = np.zeros_like(xs, dtype=float)
        for t in ts:
            ys += np.interp(xs, t.xs, t.ys)
        ys /= len(ts)

        run_names = {t.run_name for t in ts}
        if len(run_names) == 1:
            run_name = run_names.pop()
        else:
            run_name = None

        return TimeSeries(xs, ys, name or f"avg({','.join(t.name for t in ts)})", run_name)

    @staticmethod
    def max(*ts: 'TimeSeries', name: Optional[str] = None) -> 'TimeSeries':
        xs = np.unique(np.concatenate([t.xs for t in ts]))
        xs.sort()
        ys = np.zeros((len(ts), len(xs)), dtype=float)
        for i, t in enumerate(ts):
            ys[i] = np.interp(xs, t.xs, t.ys)
        ys = ys.max(axis=0)

        run_names = {t.run_name for t in ts}
        if len(run_names) == 1:
            run_name = run_names.pop()
        else:
            run_name = None

        return TimeSeries(xs, ys, name or f"max({','.join(t.name for t in ts)})", run_name)

    def __sub__(self, other: 'TimeSeries') -> 'TimeSeries':
        all_xs = np.array(list(set(self.xs) | set(other.xs)))
        all_xs.sort()
        ys = np.interp(all_xs, self.xs, self.ys) - np.interp(all_xs, other.xs, other.ys)

        run_names = {ts.run_name for ts in (self, other) if ts.run_name is not None}
        if len(run_names) == 0:
            run_name = None
        elif len(run_names) == 1:
            run_name = run_names.pop()
        else:
            run_name = f"{self.run_name} - {other.run_name}"

        return TimeSeries(
            all_xs,
            ys,
            name=f"{self.name} - {other.name}",
            run_name=run_name)


class Run:
    def __init__(self, name: Optional[str] = None):
        self.name = name

    @property
    @abstractmethod
    def available_metrics(self) -> Set[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_time_series(self, metric: str) -> TimeSeries:
        raise NotImplementedError()

    def get_average_of_time_series(self, metrics: List[str], *, name: Optional[str] = None) -> TimeSeries:
        ts = [self.get_time_series(m) for m in metrics]
        return TimeSeries.average(ts, name=name)


class WandbRun(Run):
    def __init__(
            self,
            entity: str,
            project: str,
            groups: Union[str, List[str]],
            *,
            name: Optional[str] = None,
    ):
        super().__init__(name)

        if isinstance(groups, str):
            groups = [groups]

        self.runs = []
        for group in groups:
            self.runs.extend(_wandb_api.runs(
                path=f"{entity}/{project}",
                filters={
                    "group": group
                }
            ))

    @functools.cache
    @staticmethod
    def _wandb_get_dataframe(run: wandb.apis.public.Run) -> pd.DataFrame:
        try:
            a = _wandb_api.artifact(f"{run.entity}/{run.project}/run-{run.id}-history:latest")
        except wandb.errors.CommError as e:
            if "not found" in e.message:
                # fallback to scan_history
                rows = list(run.scan_history(page_size=16 * 1024))
                return pd.DataFrame(rows)
            else:
                raise

        df = None

        for name, e in a.manifest.entries.items():
            if name.endswith(".parquet"):
                downloaded_path = os.path.join(".", "plotdevice_wandb_artifacts", run.id)
                downloaded_path = a.get_entry(name).download(downloaded_path)
                new_df = pd.read_parquet(downloaded_path)
                if df is None:
                    df = new_df
                else:
                    df = pd.concat([df, new_df])

        # It seems that running experiments only flush to artifact files occasionally, so we get the rest this way.
        if run.state == "running":
            last_step_we_have = int(max(df['_step']))
            rows = list(run.scan_history(page_size=32 * 1024, min_step=last_step_we_have+1))
            df = pd.concat([df, pd.DataFrame(rows)])

        return df

    def _get_dataframes(self) -> Iterable[pd.DataFrame]:
        return [
            df
            for df in tqdm(
                bettermap.ordered_map_per_thread(
                    WandbRun._wandb_get_dataframe,
                    self.runs,
                    parallelism=10  # connection pool size for urllib3
                ),
                total=len(self.runs))
            if df is not None
        ]

    @property
    def available_metrics(self) -> Set[str]:
        metrics = set()
        for df in self._get_dataframes():
            metrics |= set(df.columns)
        return metrics

    def get_time_series(
        self,
        metric: str,
        *,
        alternate_names: Iterable[str] = frozenset(),
    ) -> TimeSeries:
        pairs = collections.defaultdict(dict)
        all_metric_names = [metric] + list(alternate_names)
        for df in self._get_dataframes():
            for metric_name in all_metric_names:
                if metric_name not in df:
                    continue
                df = df.dropna(subset=[metric_name])
                steps = [int(s) for s in df['_step']]
                values = [float(v) for v in df[metric_name]]
                assert len(steps) == len(values)
                pairs[metric_name].update(zip(steps, values))

        # pick the first one that has data
        for metric_name in all_metric_names:
            if metric_name not in pairs:
                continue
            xs = np.array(list(pairs[metric_name].keys()))
            ys = np.array(list(pairs[metric_name].values()))
            sort_indices = np.argsort(xs)
            xs = xs[sort_indices]
            ys = ys[sort_indices]
            return TimeSeries(xs=xs, ys=ys, name=metric, run_name=self.name)

        xs = np.array([], dtype=int)
        ys = np.array([], dtype=float)
        return TimeSeries(xs=xs, ys=ys, name=metric, run_name=self.name)


class CometmlRun(Run):
    def __init__(
            self,
            workspace: str,
            project: str,
            pattern: str,
            *,
            name: Optional[str] = None,
    ):
        super().__init__(name)
        self.experiments = _cometml_api.get_experiments(workspace, project, pattern)

    @property
    def available_metrics(self) -> Set[str]:
        metrics = set()
        for experiment in self.experiments:
            metrics_summary = experiment.get_metrics_summary()
            for metric in metrics_summary:
                metrics.add(metric['name'])
        return metrics

    def get_time_series(self, metric: str) -> TimeSeries:
        xs = []
        ys = []

        for experiment in self.experiments:
            metrics = experiment.get_metric_total_df(metric)
            if metrics is None:
                metrics = experiment.get_metrics(metric=metric)
                if len(metrics) > 0:
                    # only warn if the metric does exist with the old method
                    _logger.warning(f"Can't download full fidelity metrics from CometML for {metric} in {experiment.key}. Falling back to sampled.")
                xs.extend(x['step'] for x in metrics)
                ys.extend(float(x['metricValue']) for x in metrics)
            else:
                xs.extend(metrics['step'])
                ys.extend(metrics['value'])

        assert len(xs) == len(ys)
        xs = np.array(xs)
        ys = np.array(ys)
        sort_indices = np.argsort(xs)
        xs = xs[sort_indices]
        ys = ys[sort_indices]

        return TimeSeries(xs, ys, metric, run_name=self.name)

from plotdevice import plot_with_plotly, plot_with_matplotlib
