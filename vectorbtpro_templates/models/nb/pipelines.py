from pathlib import Path
import pandas as pd
import numpy as np
import numba as nb
import vectorbtpro as vbt
import vectorbtpro._typing as tp  # -> vbt typing extension

from vectorbtpro_templates.config import param_names
from vectorbtpro_templates.models.nb.strategies import get_signals_nb


__all__ = [
    "get_portfolio_nb",
    "get_metrics_nb",
    "pipeline_nb",
    "chunked_func_nb",
    "chunked_wrapper_nb",
    "pipeline_chunked_nb",
]


# Pipeline Function for Sharpe Ratio Optimization
# -----------------------------------------------


@nb.njit(nogil=True)  # <- nogil enabled allows multithreading
def get_portfolio_nb(
    close: tp.Array1d,
    entries: tp.Array1d,
    exits: tp.Array1d,
) -> tp.NamedTuple:
    """Backtest a **single** strategy using generated entry and exit signals with Numba-compiled functions. 

    Source: https://vectorbt.pro/pvt_1606a55a/api/portfolio/nb/from_signals/#vectorbtpro.portfolio.nb.from_signals.from_signal_func_nb"""
    # Create numba compiled portfolio backtesting from signals
    # Implemented an ultrafast simulation method based on signals - from_basic_signals_nb -
    # which is used automatically if you don't use stop and limit orders
    # https://vectorbt.pro/pvt_1606a55a/getting-started/release-notes/2023/#version-1140-1-sep-2023
    return vbt.pf_nb.from_basic_signals_nb(
        target_shape=(close.shape[0], 1),
        group_lens=np.array([1]),
        auto_call_seq=True,
        close=close,
        long_entries=entries,
        long_exits=exits,
        save_returns=True,  # Pre-calculate the returns
    )


@nb.njit(nogil=True)  # <- nogil enabled allows multithreading
def get_metrics_nb(sim_out: tp.NamedTuple, ann_factor: int) -> float:
    """Generated metrics of a strategy (Sharpe ratio).

    VBT PRO provides an arsenal of Numba-compiled functions that are used by 
    accessors and for measuring portfolio performance. These only accept NumPy 
    arrays and other Numba-compatible types.

    Source: https://vectorbt.pro/pvt_1606a55a/api/returns/nb/#vectorbtpro.returns.nb.sharpe_ratio_1d_nb"""
    # Numba-compiled version of the analysis part.
    # https://vectorbt.pro/pvt_1606a55a/tutorials/pairs-trading/#level-engineer
    # https://vectorbt.pro/pvt_1606a55a/tutorials/cross-validation/applications/#decorators
    # More metrics: https://vectorbt.pro/pvt_1606a55a/api/returns/nb/
    returns = sim_out.in_outputs.returns
    # Note on "ddof":
    # - Use ddof=0 for population variance, where you have data for the entire population.
    # - Use ddof=1 for sample variance, to correct for bias when estimating the variance of a population from a sample.
    sharpes = vbt.ret_nb.sharpe_ratio_nb(returns, ann_factor, ddof=1)[0]
    # TODO: See sharpe_ratio_1d_nb to return a float directly?
    # https://vectorbt.pro/pvt_1606a55a/api/returns/nb/#vectorbtpro.returns.nb.sharpe_ratio_1d_nb
    return sharpes


@nb.njit(nogil=True)  # <- nogil enabled allows multithreading
def pipeline_nb(
    close: tp.Array1d,
    fastperiod: int,
    slowperiod: int,
    signalperiod: int,
    timeperiod: int,
    window: int,
    alpha: float,
    ann_factor: int, 
) -> float:
    """Backtest a **single** strategy and calculate metric sepcified in `get_metric_nb`."""
    # Create signals with specified parameters
    entries, exits = get_signals_nb(close, fastperiod, slowperiod, signalperiod, timeperiod, window, alpha)
    # Create a portfolio from signals_nb, a numba compiled function.
    sim_out = get_portfolio_nb(close, entries, exits)
    # Get metrics
    return get_metrics_nb(sim_out, ann_factor)


@nb.njit(nogil=True)  # <- nogil enabled allows multithreading
def chunked_func_nb(
    n_params: int,
    close: tp.Array1d,
    fastperiod: tp.FlexArray1dLike,
    slowperiod: tp.FlexArray1dLike,
    signalperiod: tp.FlexArray1dLike,
    timeperiod: tp.FlexArray1dLike,
    window: tp.FlexArray1dLike,
    alpha: tp.FlexArray1dLike,
    ann_factor: int
) -> tp.Array1d:
    """Backtest **multiple** strategies using generated entry and exit signals with Numba-compiled functions.

    Returns metric sepcify in `get_metric_nb`."""
    # Numba pipeline:
    # https://vectorbt.pro/pvt_1606a55a/tutorials/superfast-supertrend/pipelines/#numba-pipeline
    # The function that we will use below requires the parameters to be
    # one-dimensional NumPy arrays, thus, convert each one to such an array in
    # the case it isn't such yet. We must write the result to a new variable.
    fastperiod_ = vbt.to_1d_array_nb(np.asarray(fastperiod))
    slowperiod_ = vbt.to_1d_array_nb(np.asarray(slowperiod))
    signalperiod_ = vbt.to_1d_array_nb(np.asarray(signalperiod))
    timeperiod_ = vbt.to_1d_array_nb(np.asarray(timeperiod))
    window_ = vbt.to_1d_array_nb(np.asarray(window))
    alpha_ = vbt.to_1d_array_nb(np.asarray(alpha))

    # Create empty NumPy arrays that we want to return - we will gradually fill
    # them in the loop below
    metrics = np.empty(n_params, dtype=vbt.float_)
    # other_metrics = np.empty(n_params, dtype=vbt.float_)

    for i in range(n_params):
        metrics[i] = pipeline_nb(
        # metrics[i], other_metrics[i] = pipeline_nb(
            close,
            fastperiod=vbt.flex_select_1d_nb(fastperiod_, i),
            slowperiod=vbt.flex_select_1d_nb(slowperiod_, i),
            signalperiod=vbt.flex_select_1d_nb(signalperiod_, i),
            timeperiod=vbt.flex_select_1d_nb(timeperiod_, i),
            window=vbt.flex_select_1d_nb(window_, i),
            alpha=vbt.flex_select_1d_nb(alpha_, i),
            ann_factor=ann_factor
        )
    return metrics


# Split pipeline into chunks
chunked_wrapper_nb = vbt.chunked(
    chunked_func_nb,
    size=vbt.ArgSizer(arg_query="n_params"),
    arg_take_spec=dict(
        n_params=vbt.CountAdapter(),
        close=None,  # <- Set arguments that should be passed without chunking to None
        ann_factor=None,
        **{name: vbt.FlexArraySlicer() for name in param_names}
        # window=vbt.FlexArraySlicer(),
        # alpha=vbt.FlexArraySlicer(),
        # etc ...
    ),
    chunk_len='auto', # <- Number of parameter combinations to process during each iteration of the while-loop.
    # Apart from chunking the parameter arrays, we can also put chunks themselves into so-called "super chunks". 
    # Each super chunk will consist of as many chunks as there are CPU cores - one per thread.
    merge_func="concat",
    # Execution
    # -> Processes:
    # execute_kwargs=dict(n_chunks="auto", distribute="chunks", engine="pathos"),
    # -> Super-Chunks
    execute_kwargs=dict(chunk_len="auto", engine="threadpool"),
    # Each super chunk will consist of as many chunks as there are CPU cores - one per thread.
    # https://vectorbt.pro/pvt_1606a55a/cookbook/optimization/#hybrid-super-chunks
    # Any argument passed to the chunked decorator can be overridden
    # during the runtime using the same argument but prefixed with
    # an underscore _
)
"""Wrap `chunked_func_nb` with the @chunked decorator.

Source: https://vectorbt.pro/pvt_1606a55a/tutorials/superfast-supertrend/pipelines/#chunked-pipeline"""


def pipeline_chunked_nb(
    close: tp.Array1d,
    params: tp.Dict[str, vbt.Param],
    ann_factor: int,
    path: tp.Optional[str | Path] = None,
    to_pd_series: tp.Optional[bool] = False,
    **exe_kwargs
) -> tp.Array1d | pd.Series:
    """Backtest **multiple** strategies into chunks.

    Returns metric arraysepcify in `get_metric_nb`."""
    # Construct the parameter grid manually
    param_product, param_index = vbt.combine_params(params)
    # Total number of parameter combinations
    n_params = len(param_index)
    # Extract kwargs
    merged_kwargs = vbt.merge_dicts(
        param_product,
        dict(n_params=n_params, close=close, ann_factor=ann_factor, **exe_kwargs)
    )
    if path is not None and vbt.file_exists(path):
        metrics = vbt.load(path)
    else:
        # Iterate over chunks and pass each subset to the parent function for execution
        metrics = chunked_wrapper_nb(**merged_kwargs)
        # Save the result if path is provided
        if path is not None:
            vbt.save(metrics, path)

    return pd.Series(metrics, index=param_index) if to_pd_series else metrics
