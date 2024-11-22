import numba as nb
import vectorbtpro as vbt
import vectorbtpro._typing as tp  # -> vbt typing extension

__all__ = ["get_signals"]


# Custom Signal Generation Function
# ---------------------------------


@nb.njit(nogil=True)  # <- nogil enabled allows multithreading
def strategy_nb(
    close: tp.Array1d,
    macd: tp.Array1d,
    signal: tp.Array1d,
    rsi: tp.Array1d,
    upperband: tp.Array1d,
    lowerband: tp.Array1d
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Defines entry and exit signals using MACD, RSI, and Bollinger Bands. 

    The function is optimized with Numba for performance.
    Source: https://vectorbt.pro/pvt_1606a55a/api/indicators/factory/"""
    # Note: Numba compiled **alternative** via vbt: all_reduce_nb(arr)
    # https://vectorbt.pro/pvt_1606a55a/api/generic/nb/apply_reduce/#vectorbtpro.generic.nb.apply_reduce.all_reduce_nb
    entries = (
        (macd >= signal) &
        (rsi < 30) &
        (lowerband > close)  # Or vbt.nb.crossed_below_nb?
    )
    exits = (
        (macd < signal) &
        (rsi > 70) &
        (upperband < close)  # Or vbt.nb.crossed_above_nb?
    )

    return entries, exits


# Custom Indicator Functions
# --------------------------


def get_signals(
    close: tp.Array1d,
    fastperiod: int,
    slowperiod: int,
    signalperiod: int,
    timeperiod: int,
    window: int,
    alpha: float
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Generate MACD, RSI, and Bollinger Bands indicators using TA-Lib.

    Source: https://vectorbt.pro/pvt_1606a55a/tutorials/superfast-supertrend/#numpy-numba-talib"""
    macd, signal, _ = (
        vbt.talib_func("MACD")
        (close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    )
    rsi = (
        vbt.talib_func("RSI")
        (close, timeperiod=timeperiod)
    )
    upperband, _, lowerband = (
        vbt.talib_func("BBANDS")
        (close, timeperiod=window, nbdevup=alpha, nbdevdn=alpha)
    )
    return strategy_nb(close, macd, signal, rsi, upperband, lowerband)
