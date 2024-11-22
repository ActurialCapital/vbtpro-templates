import numpy as np
import numba as nb
import vectorbtpro._typing as tp  # -> vbt typing extension

__all__ = ["np_list_arange"]


@nb.njit
def np_list_arange(
    start: int | float,
    stop: int | float,
    step: int | float,
    inclusive: tp.Optional[bool] = False
) -> tp.List[int | float]:
    """
    Generates a list of values within a specified range with step intervals,
    similar to numpy.arange but returns a list. It also ensures correct type 
    handling for floating point values.

    Parameters
    ----------
    start : int | float
        Starting value of the sequence.
    stop : int | float
        End value of the sequence.
    step : int | float
        Step size between consecutive numbers.
    inclusive : bool, optional
        If True, `stop` is the last value in the range, by default False.

    Returns
    -------
    tp.List[int | float]
        A list of values from `start` to `stop` with `step` intervals.

    Examples
    --------
    >>> print(np_list_arange(1, 5, 1))
    [1, 2, 3, 4]
    >>> print(np_list_arange(1, 5, 1, inclusive=True))
    [1, 2, 3, 4, 5]
    """
    convert_to_float = (
        isinstance(start, float) or
        isinstance(stop, float) or
        isinstance(step, float)
    )
    if convert_to_float:
        stop = float(stop)
        start = float(start)
        step = float(step)
    stop = stop + (step if inclusive else 0)
    range_ = list(np.arange(start, stop, step))
    range_ = [
        start
        if x < start
        else stop
        if x > stop
        else float(round(x, 15))
        if isinstance(x, float)
        else x
        for x in range_
    ]
    range_[0] = start
    range_[-1] = stop - step
    return range_
