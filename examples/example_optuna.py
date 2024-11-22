from pathlib import Path
import optuna
import vectorbtpro as vbt

from vectorbtpro_templates import (
    get_data_from_csv,
    optuna_objective_talib,
    optuna_objective_nb,
    default_optuna_study,
    default_optuna_optimize
)

try:
    DATA_DIR = Path(__file__).resolve().parent
except:
    pass


if __name__ == "__main__":

    # Load historical data from CSV
    path = DATA_DIR / "csv" / "NQ=F_ohlcv_data.csv"
    data = get_data_from_csv(path, sep=";")
    close = vbt.to_1d_array(data.close)

    # Instead of testing the full parameter grid, we can adopt a statistical approach.
    # There are libraries, such as Hyperopt or Optuna, that are tailored at
    # minimizing objective functions.

    # With TA-Lib

    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        study = optuna.create_study(**default_optuna_study)
        study.optimize(optuna_objective_talib(data), **default_optuna_optimize)

    print('Run Optuna Implementation')
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())
    # Run Optuna Implementation
    # Time elapsed: 28.76 seconds
    # Memory usage: 31.8 MB

    print(study.trials_dataframe())
    print(study.best_params)

    # With Numba-compiled

    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        study = optuna.create_study(**default_optuna_study)
        study.optimize(optuna_objective_nb(close), **default_optuna_optimize)

    print('Run Optuna Implementation')
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())
    # Run Optuna Implementation
    # Time elapsed: 1 minute and 29.95 seconds
    # Memory usage: 362.8 MB

    print(study.trials_dataframe())
    print(study.best_params)
