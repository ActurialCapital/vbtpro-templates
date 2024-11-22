# VectorBT PRO Templates

A repository containing advanced templates and tools for working with VectorBT PRO, enabling efficient backtesting and parameter optimization.

## Requirements

### Authentication

#### Option 1: Token

After you've been added to the list of collaborators and accepted the repository invitation, the next step is to create a [Personal Access Token](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token) for your GitHub account in order to access the PRO repository programmatically (from the command line or GitHub Actions workflows):

1. Go to https://github.com/settings/tokens
2. Click on [Generate a new token (classic)]
3. Enter a name (such as "terminal")
4. Set the expiration to some fixed number of days
5. Select the [repo](https://docs.github.com/en/developers/apps/scopes-for-oauth-apps#available-scopes) scope
6. Generate the token and save it in a safe place

#### Option 2: Credential Manager

Alternatively, use [Git Credential Manager](https://github.com/git-ecosystem/git-credential-manager) instead of creating a personal access token.

## Installation

### Clone the Repository

Clone the repository using the web URL:

```sh
$ git clone https://github.com/ActurialCapital/vbtpro-templates.git
```

### Set Up the Environment

We recommend using Poetry for dependency management and virtual environment creation. Run the following command to install dependencies, generate a lock file, and activate the virtual environment:

```sh
$ poetry install && poetry lock && poetry shell
```

> [!NOTE]
> If you plan to use vectorbtpro locally, it's recommended to establish a new environment solely for `vectorbtpro`. 

### Install VectorBT PRO

Use the following command to install the PRO version of vectorbtpro:

- Replace [GH_USER] with your GitHub username.
- Replace [GH_TOKEN] with your Personal Access Token.

```sh
$ poetry run pip3 install -U "vectorbtpro[base] @ git+https://[GH_USER]:[GH_TOKEN]@github.com/polakowo/vectorbt.pro.git"
```

## Run the Application

1. Navigate to the `examples` directory:

```sh
$ cd ./examples
```

2. Run the script using Poetry:

```sh
$ poetry run python run_app.py
```

Example output during parameter combination processing:

```less
[INFO] Starting parameter combination process...
[INFO] Time elapsed for combine_params: 12.75 seconds
[INFO] Total number of parameter combinations: 14,400,000
[INFO] Chunk size: 100,000
[INFO] Number of chunks [CPU cores=8]: 18
[INFO] Processing chunks...
100%|█████████████████████████████████████████████████| 18/18 [2:22:11<00:00, 473.96s/it, chunk_tasks=136..143]
[INFO] Overall Progress:
[INFO] Time elapsed: 2 hours, 22 minutes and 12.15 seconds
[INFO] Memory usage: 6.4 GB
```

3. Key Performance

- **Parameter Combination Process**: `14,400,000`
- **Chunking Strategy**: Parameters are split into chunks of `100,000` combinations each, allowing parallel processing across `8` CPU cores
- **Engine**: `threadpool`
- **Estimated time per chunk**: `~474 seconds` (or `7 minutes and 54 seconds`)
- **Estimated time per combination**: `~0.000592 seconds` (or `592 microseconds`)
- **Overall time spent**: `~2 hours and 22 minutes`
- **Resource Utilization**: `6.4 GB`
  
## Features

### Parameter Optimization

- **Grid Search**: Test full parameter grids efficiently using parallel processing.
- **Chunking Strategy**: Automatically splits large grids into manageable chunks for optimal performance.
- **Hyperparameter Tuning**: Integrate with tools like Optuna for advanced optimization.

### Custom Indicators and Strategies

- Easily create custom technical indicators with Numba's indicator factory.
- Define complex entry/exit signals for portfolio strategies using compiled functions.

## Getting Started

### Dependencies

Import VectorBT PRO and `vectorbtpro_templates`:

```python
>>> import vectorbtpro as vbt
>>> from vectorbtpro_templates import *
```

### Configuration Templates

The following predefined configurations are used throughout the examples to simplify and standardize parameter setups for portfolios and technical indicators:

- **Default Single Parameter** (`default_single_params`): This template contains a single set of parameters, designed for quick testing and demonstration of individual strategies. It is ideal for running a single backtest with fixed inputs.
- **Default Parameters** (`default_params`): A flexible configuration that supports multiple sets of parameters for testing and analysis. These parameters are often used to define a broader range of inputs for more complex simulations or optimizations.
- **Parameterized** (`default_vbt_params`): This configuration is structured to facilitate full parameter grid exploration. It is designed to work seamlessly with vectorbt's `@parameterized` decorator or `@chunked`-based processing, allowing for extensive testing across multiple parameter combinations.

These templates ensure consistency and make it easier to replicate, modify, or extend configurations across different strategies. By using them, you can focus on understanding results and tuning strategies rather than setting up parameters repeatedly.

### Load Data

Load historical data from CSV:

```python
>>> path = "path/to/your/data.csv"
>>> data = get_data_from_csv(path, sep=";")
>>> # Get the annualization factor to calculate sharpe:
>>> ann_factor = int(vbt.pd_acc.returns.get_ann_factor(freq='D'))
>>> # Get close prices in a NumPy array
>>> close = vbt.to_1d_array(data.close)
# array([ 3653.5,  3804. ,  3853. , ..., 19496.5, 19602.5, 19685.5])
```

### Custom Indicator

Module with custom indicators built with the indicator factory:

```python
>>> st_nb = StrategyNumba.run(
...     data.close,
...     param_product=True,
...     execute_kwargs=dict(n_chunks="auto", distribute="chunks", engine="pathos"),
...     **default_params
... )
```

### Backtest a Single Strategy

```python
>>> pipeline_nb(close, ann_factor=ann_factor, **default_single_params)
```

### Run Parameterized Strategies

#### Option 1: `@parametrized`

```python
>>> # Test full parameter grid (automatic parameter combinations)
>>> pipe = vbt.parameterized(
...     pipeline_nb,
...     merge_func="concat",
...     engine="pathos", distribute="chunks", n_chunks="auto"
... )
>>> sharpes = pipe(close, ann_factor=ann_factor, **default_vbt_params)
```

> [!NOTE]
> Instead of testing the full parameter grid, we can adopt a statistical approach at this stage.
> Control the number of random trials with `random_subset`.

> [!WARNING]
> The parameterized decorator has two major limitations: it runs only one parameter combination at a time, and it needs to build the parameter grid fully, even when querying a random subset.

#### Option 2: Super-`@chunked`

1. Construct the parameter grid manually
2. Get the total number of parameter combinations to process during each iteration of the while-loop
3. Distribute parameter combinations into chunks of an optimal length, and execute all parameter combinations within each chunk in parallel with multithreading (i.e., one parameter combination per thread) while executing chunks themselves serially

```python
>>> # Test full parameter grid (built-in automatic parameter combinations)
>>> sharpes = pipeline_chunked_nb(close, default_vbt_params, ann_factor=ann_factor)
```

> [!IMPORTANT]
> Threads are easier and faster to spawn than processes. Also, to execute a function in its own process, all the passed inputs and parameters need to be serialized and then deserialized, which takes time. Thus, multithreading is preferred, but it requires the function to release the GIL, which means either compiling the function with Numba and setting the nogil flag to True, or using exclusively NumPy.

### Hyperparameter Tuning (Bonus)

Instead of testing the full parameter grid, we can adopt a statistical approach. There are libraries, such as `Hyperopt` or `Optuna`, that are tailored at minimizing (maximizing) objective functions.

> [Optuna](https://optuna.org/) is framework agnostic. You can use it with any machine learning or deep learning framework. It is extensively used for hyperparamter optimization.

```python
>>> import optuna
>>> study = optuna.create_study(**default_optuna_study)
>>> study.optimize(optuna_objective_talib(data), **default_optuna_optimize)
>>> # Output study in pandas DataFrame object
>>> sharpes = study.trials_dataframe()
```

## Tutorial

This tutorial demonstrates a signal-generation strategy using MACD, RSI, and BBANDS indicators:

- **Entries**:
  - `MACD LINE >= MACD SIGNAL`
  - `RSI < 30`
  - `LOWER BBANDS > CLOSE`

- **Exits**:
  - `MACD LINE < MACD SIGNAL`
  - `RSI > 70`
  - `UPPER BBANDS < CLOSE`

The implementation and configuration files are located in:
- `vectorbtpro_templates/models/nb/strategies`
- `vectorbtpro_templates/config`

> [!NOTE]
> All functions are optimized with Numba for performance:
> ```python
> import numba as nb
> ```

### Strategy

1. **Define your signals** (conditions): 

```python
>>> @nb.njit(nogil=True)  # <- nogil enabled allows multithreading
>>> def strategy_nb(
...     close: tp.Array1d,
...     macd: tp.Array1d,
...     signal: tp.Array1d,
...     rsi: tp.Array1d,
...     upperband: tp.Array1d,
...     lowerband: tp.Array1d
... ) -> tp.Tuple[tp.Array1d, tp.Array1d]:
...         entries = (
...             (macd >= signal) &
...             (rsi < 30) &
...             (lowerband > close) 
...         )
...         exits = (
...             (macd < signal) &
...             (rsi > 70) &
...             (upperband < close) 
...         )
... return entries, exits
```

2. **Generate indicators and return signals**:

```python
>>> @nb.njit(nogil=True)  # <- nogil enabled allows multithreading
>>> def get_signals_nb(
...     close: tp.Array1d,
...     fastperiod: int,
...     slowperiod: int,
...     signalperiod: int,
...     timeperiod: int,
...     window: int,
...     alpha: float
... ) -> tp.Tuple[tp.Array1d, tp.Array1d]:
...     macd, signal = (
...         vbt.indicators.nb.macd_1d_nb  # <- single asset
...         (close, fast_window=fastperiod, slow_window=slowperiod, signal_window=signalperiod)
...     )
...     rsi = (
...         vbt.indicators.nb.rsi_1d_nb  # <- single asset
...         (close, window=timeperiod)
...     )
...     upperband, _, lowerband = (
...         vbt.indicators.nb.bbands_1d_nb  # <- single asset
...         (close, window=window, alpha=alpha)
...     )
...     return strategy_nb(close, macd, signal, rsi, upperband, lowerband)
```

### Pipelines

Pipelines are located in `vectorbtpro_templates/models/nb/pipelines`.

1. **Simulator**: 

```python
>>> @nb.njit(nogil=True)  # <- nogil enabled allows multithreading
>>> def get_portfolio_nb(
...     close: tp.Array1d,
...     entries: tp.Array1d,
...     exits: tp.Array1d,
... ) -> tp.NamedTuple:
...     # Create numba compiled portfolio backtesting from signals
...     # Implemented an ultrafast simulation method based on signals 
...     # `from_basic_signals_nb`.
...     return vbt.pf_nb.from_basic_signals_nb(
...         target_shape=(close.shape[0], 1),
...         group_lens=np.array([1]),
...         auto_call_seq=True,
...         close=close,
...         long_entries=entries,
...         long_exits=exits,
...         save_returns=True,  # Pre-calculate the returns
...     )
```

2. **Generated metric(s)** (Sharpe ratio):

> [!NOTE]
> VectorBT PRO provides an arsenal of Numba-compiled functions that are used by accessors and for measuring portfolio performance. These only accept NumPy arrays and other Numba-compatible types.

```python
>>> @nb.njit(nogil=True)  # <- nogil enabled allows multithreading
>>> def get_metrics_nb(sim_out: tp.NamedTuple, ann_factor: int) -> float:
...     returns = sim_out.in_outputs.returns
...     # Note on "ddof":
...     # - Use ddof=0 for population variance, where you have data for the entire population.
...     # - Use ddof=1 for sample variance, to correct for bias when estimating the variance of a population from a sample.
...     sharpes = vbt.ret_nb.sharpe_ratio_nb(returns, ann_factor, ddof=1)[0] # -> Extract float
...     return sharpes
```

3. Backtest a **single** strategy and calculate a metric sepcified in `get_metric_nb`:

```python
>>> @nb.njit(nogil=True)  # <- nogil enabled allows multithreading
>>> def pipeline_nb(
...     close: tp.Array1d,
...     fastperiod: int,
...     slowperiod: int,
...     signalperiod: int,
...     timeperiod: int,
...     window: int,
...     alpha: float,
...     ann_factor: int
... ) -> float:
...     # Create signals with specified parameters
...     # Create a portfolio from `signals_nb`.
...     entries, exits = get_signals_nb(close, fastperiod, slowperiod, signalperiod, timeperiod, window, alpha)
...     sim_out = get_portfolio_nb(close, entries, exits)
...     # Get metrics
...     return get_metrics_nb(sim_out, ann_factor)
```

4. Run parametrized strategies (cartesian product) using super-`@chunked`:
   - Construct the parameter grid manually.
   - Get the total number of parameter combinations to process during each iteration of the while-loop.
   - Distribute parameter combinations into chunks of an (optimal) length, and execute all parameter combinations within each chunk in parallel with multithreading (i.e., one parameter combination per thread) while executing chunks themselves serially.
   - Optionally save/load file from disk.
   - Optionally output a `pandas.Series` object.

```python
>>> def pipeline_chunked_nb(
...     close: tp.Array1d,
...     params: tp.Dict[str, vbt.Param],
...     ann_factor: int,
...     path: tp.Optional[str | Path] = None,
...     to_pd_series: tp.Optional[bool] = False,
...     **exe_kwargs
... ) -> tp.Array1d | pd.Series:
...     # Construct the parameter grid manually
...     param_product, param_index = vbt.combine_params(params)
...     # Total number of parameter combinations
...     n_params = len(param_index)
...     # Extract kwargs
...     merged_kwargs = vbt.merge_dicts(
...         param_product,
...         dict(n_params=n_params, close=close, ann_factor=ann_factor, **exe_kwargs)
...     )
...     if path is not None and vbt.file_exists(path):
...         metrics = vbt.load(path)
...     else:
...         # Iterate over chunks and pass each subset to the parent function for execution
...         metrics = chunked_wrapper_nb(**merged_kwargs)
...         # Save the result if path is provided
...         if path is not None:
...             vbt.save(metrics, path)
...     # Returns a pandas series or a numpy array
...     return pd.Series(metrics, index=param_index) if to_pd_series else metrics
```

## Support

If you'd like to further contribute and help fuel the ongoing development and improvements of my work, you can buy me a coffee! ☕
Thank you again for your partnership and support—I look forward to working with you on future projects!

<a href="https://www.buymeacoffee.com/acturialcapital"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=skforecast&button_colour=f79939&font_colour=000000&font_family=Poppins&outline_colour=000000&coffee_colour=FFDD00" /></a>


