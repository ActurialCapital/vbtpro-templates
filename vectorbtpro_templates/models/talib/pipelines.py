import vectorbtpro as vbt

from vectorbtpro_templates.models.talib.custom_indicators import StrategyTALib
from vectorbtpro_templates.config import default_port_kwargs


__all__ = ["pipeline_talib"]

# Pipeline Function for Sharpe Ratio Optimization
# -----------------------------------------------


def pipeline_talib(
    data: vbt.Data,
    fastperiod: int,
    slowperiod: int,
    signalperiod: int,
    timeperiod: int,
    window: int,
    alpha: float,
) -> float:
    """Backtest single/multiple strategy(ies) using generated entry and exit signals with TA-Lib.

    Returns Sharpe ratio."""
    # try:
    # Run custom indicator with specified parameters
    st = StrategyTALib.run(
        data.close,
        fastperiod, slowperiod, signalperiod, timeperiod, window, alpha,
        return_raw=True
    )
    # Create a portfolio from signals and calculate the Sharpe ratio
    pf = vbt.Portfolio.from_signals(
        data,
        entries=st[0][0],
        exits=st[0][1],
        **default_port_kwargs
    )
    return pf.sharpe_ratio
    # except Exception:
    #     return vbt.NoResult
