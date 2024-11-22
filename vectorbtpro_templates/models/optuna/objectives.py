import numpy as np
import optuna
import vectorbtpro as vbt
import vectorbtpro._typing as tp  # -> vbt typing extension

from vectorbtpro_templates.models.talib.pipelines import pipeline_talib
from vectorbtpro_templates.models.nb.pipelines import pipeline_nb

__all__ = ["optuna_objective_talib", "optuna_objective_nb"]

# Disable Optuna logging entirely
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


# Hyperparameter optimization using Optuna
# ---------------------------------------------------
# Optuna is framework agnostic. You can use it with any machine learning or deep learning framework.
# It is extensively used for hyperparamter optimization.
# https://optuna.org/

def optuna_objective_talib(data: vbt.Data):
    def objective(trial: optuna.Trial) -> float:
        """Maximize sharpe ratio using TA-Lib."""
        metric = pipeline_talib(
            data,
            fastperiod=trial.suggest_int('fastperiod', 5, 15),
            slowperiod=trial.suggest_int('slowperiod', 20, 30),
            signalperiod=trial.suggest_int('signalperiod', 3, 13),
            timeperiod=trial.suggest_int('timeperiod', 2, 12),
            window=trial.suggest_int('window', 5, 10),
            alpha=trial.suggest_float('alpha', 0.5, 2.3, step=0.2),
        )
        if np.isnan(metric):
            raise optuna.TrialPruned()
            # See: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.TrialPruned.html

        return metric
    return objective


def optuna_objective_nb(close: tp.Array1d):
    def objective(trial: optuna.Trial) -> float:
        """Maximize sharpe ratio using Number-compiled functions."""
        metric = pipeline_nb(
            close,
            fastperiod=trial.suggest_int('fastperiod', 5, 15),
            slowperiod=trial.suggest_int('slowperiod', 20, 30),
            signalperiod=trial.suggest_int('signalperiod', 3, 13),
            timeperiod=trial.suggest_int('timeperiod', 2, 12),
            window=trial.suggest_int('window', 5, 10),
            alpha=trial.suggest_float('alpha', 0.5, 2.3, step=0.2),
            ann_factor=vbt.pd_acc.returns.get_ann_factor(freq='D')
        )
        if np.isnan(metric):
            raise optuna.TrialPruned()
            # See: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.TrialPruned.html

        return metric
    return objective
