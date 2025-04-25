import dataclasses
import functools
import logging
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
        xs = np.interp(self.xs, x_mapper.xs, x_mapper.ys)
        return TimeSeries(xs, ys, name or f'{self.name} vs {x_mapper.name})', self.run_name)

    def smooth_with_moving_average(self, width: int = 100, *, name: Optional[str] = None) -> 'TimeSeries':
        xnew = np.linspace(self.xs[0], self.xs[-1], num=len(self.xs))
        ynew = np.interp(xnew, self.xs, self.ys)
        ynew = _moving_average(ynew, width)
        xnew = xnew[width // 2:]
        xnew = xnew[:len(ynew)]
        return TimeSeries(xnew, ynew, name or self.name, self.run_name)

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
                return pd.DataFrame()
            else:
                raise
        return pd.read_parquet(a.file())

    def _get_dataframes(self) -> Iterable[pd.DataFrame]:
        return tqdm(bettermap.ordered_map_per_thread(WandbRun._wandb_get_dataframe, self.runs), total=len(self.runs))

    @property
    def available_metrics(self) -> Set[str]:
        metrics = set()
        for df in self._get_dataframes():
            metrics |= set(df.columns)
        return metrics

    def get_time_series(self, metric: str) -> TimeSeries:
        xs = []
        ys = []
        for df in self._get_dataframes():
            if metric not in df:
                continue
            df = df.dropna(subset=[metric])
            steps = [int(s) for s in df['_step']]
            values = [float(v) for v in df[metric]]
            assert len(steps) == len(values)
            xs.extend(steps)
            ys.extend(values)

        xs = np.array(xs)
        ys = np.array(ys)
        sort_indices = np.argsort(xs)
        xs = xs[sort_indices]
        ys = ys[sort_indices]

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
                _logger.warning(f"Can't download full fidelity metrics from CometML for {metric} in {experiment.key}. Falling back to sampled.")
                metrics = experiment.get_metrics(metric=metric)
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
