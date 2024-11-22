import vectorbtpro as vbt

from vectorbtpro_templates.models.talib.strategies import get_signals
from vectorbtpro_templates.config import default_single_params

__all__ = ["StrategyTALib"]


# Strategy Configuration with Indicator Factory
# ---------------------------------------------


StrategyTALib = vbt.IF(
    class_name='StrategyTALib',
    short_name='st',
    input_names=['close'],
    param_names=list(default_single_params),
    output_names=['entries', 'exits']
).with_apply_func(
    get_signals,  # <- TA-Lib
    takes_1d=True,  # <- Single asset
    execute_kwargs=dict(engine='threadpool', chunk_len='auto'),
    **default_single_params
)
"""Defines a custom indicator using the vbt `IndicatorFactory` using TA-Lib, enabling parameterized optimization.

Source: https://vectorbt.pro/pvt_1606a55a/tutorials/superfast-supertrend/#indicator-factory"""
