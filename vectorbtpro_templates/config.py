import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import vectorbtpro as vbt
import vectorbtpro._typing as tp  # -> vbt typing extension

from vectorbtpro_templates.utils import np_list_arange


# Set seed for Optuna sampler
SEED = 1234


# Configuration Classes
# ---------------------
# These configurations define parameters for the portfolio and technical indicators.
# The `NamedTuple` structure (optional) aligns with vectorbt.pro requirements for configuration classes.
# https://vectorbt.pro/pvt_1606a55a/api/generic/enums/


# MAIN PARAMETER TEMPLATE
# #######################

class ParamTemplate(tp.NamedTuple):
    # MACD
    fastperiod: tp.Iterable[int] | int
    slowperiod: tp.Iterable[int] | int
    signalperiod: tp.Iterable[int] | int
    """Defines parameters for the Moving Average Convergence Divergence (MACD) indicator."""

    # RSI
    timeperiod: tp.Iterable[int] | int
    """Defines parameters for the Relative Strength Index (RSI) indicator."""

    # BBANDS
    window: tp.Iterable[int] | int
    alpha: tp.Iterable[float] | float
    """Defines parameters for Bollinger Bands (BBands) indicator."""


# OPTIONAL (DEFAULT) TEMPLATES
# ############################

class PortConfig(tp.NamedTuple):
    """Defines portfolio parameters for backtesting the strategy."""
    # https://vectorbt.pro/pvt_12537e02/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio.from_signals
    init_cash: int = 1_000_000
    size: int = 1_000
    fees: float = 0.0
    size_type: str = 'amount'
    cash_sharing: bool = True
    freq = 'auto'
    # ...


class OptunaStudy(tp.NamedTuple):
    """Defines Optuna study parameters."""
    study_name: str = 'strategy'
    pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner()
    # -> Other Optuna pruner options:
    # - pruners.HyperbandPruner()
    # - pruners.MedianPruner()
    # - pruners.NopPruner()
    sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(
        seed=SEED)
    # -> Other Optuna sampler options:
    # - samplers.RandomSampler(seed=SEED)
    # - None
    direction: str = 'maximize'  # sharpe ratio (the higher the better)


class OptunaOptimze(tp.NamedTuple):
    """Defines Optuna optimize parameters."""
    n_trials: int = 500
    n_jobs: int = -1
    callbacks: tp.List[tp.Any] = [
        MaxTrialsCallback(100, states=(TrialState.COMPLETE,))]
    # While the n_trials argument sets the number of trials that will be run,
    # you may want to continue running until you have a certain number of successfully
    # completed trials or stop the study when you have a certain number of trials that
    # fail. This MaxTrialsCallback class allows you to set a maximum number of trials
    # for a particular TrialState before stopping the study.
    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.MaxTrialsCallback.html


# Parameter Names
# ---------------


param_names = list(ParamTemplate._fields)


# Default Parameters (HELPERS)
# ----------------------------

# Combine default settings for the portfolio and indicators into a dictionary.

default_port_kwargs = PortConfig()._asdict()

default_single_params = ParamTemplate(
    fastperiod=13,
    slowperiod=27,
    signalperiod=4,
    timeperiod=3,
    window=7,
    alpha=2.3)._asdict()

default_params = ParamTemplate(
    fastperiod=range(5, 8),
    slowperiod=range(20, 23),
    signalperiod=range(3, 6),
    timeperiod=range(2, 5),
    window=range(5, 8),
    alpha=np_list_arange(0.5, 1.0, 0.2))._asdict()

default_vbt_params = {key: vbt.Param(value) for key, value in default_params.items()}

default_optuna_study = OptunaStudy()._asdict()

default_optuna_optimize = OptunaOptimze()._asdict()
