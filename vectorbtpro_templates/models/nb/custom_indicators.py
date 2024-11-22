import vectorbtpro as vbt

from vectorbtpro_templates.models.nb.strategies import get_signals_nb
from vectorbtpro_templates.config import param_names

__all__ = ["StrategyNumba"]


# Strategy Configuration with Indicator Factory
# ---------------------------------------------


StrategyNumba = vbt.IF(
    class_name='StrategyNumba',
    short_name='st_nb',
    input_names=['close'],
    param_names=param_names,
    output_names=['entries', 'exits']
).with_apply_func(
    get_signals_nb,  # <- Numba
    takes_1d=True,  # <- Single asset
    # ... No default args (optional)
)
"""Defines a strategy using the vbt `IndicatorFactory` using Numba-compiled functions for custom indicators, enabling parameterized optimization.

Source: https://vectorbt.pro/pvt_1606a55a/tutorials/superfast-supertrend/#indicator-factory"""
