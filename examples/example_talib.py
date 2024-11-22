from pathlib import Path
import vectorbtpro as vbt

from vectorbtpro_templates import get_data_from_csv, StrategyTALib, default_params

try:
    DATA_DIR = Path(__file__).resolve().parent
except:
    pass


if __name__ == "__main__":

    # Load historical data from CSV
    path = DATA_DIR / "csv" / "NQ=F_ohlcv_data.csv"
    data = get_data_from_csv(path, sep=";")
    close = vbt.to_1d_array(data.close)

    # Run custom indicator
    # Execution tips https://vectorbt.pro/pvt_1606a55a/cookbook/optimization/#execution
    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        st = StrategyTALib.run(
            data.close,
            param_product=True,
            execute_kwargs=dict(chunk_len="auto", engine="threadpool"),
            **default_params
        )
    print("Run TA-Lib Custom Indicator")
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())
    # Run TA-Lib Custom Indicator
    # Time elapsed: 2.59 seconds
    # Memory usage: 21.1 MB

    # Run backtest
    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=st.entries,
            exits=st.exits,
            freq='D'
        )
        sharpes = pf.sharpe_ratio

    print("Run TA-Lib Backtest")
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())
    # Run TA-Lib Backtest
    # Time elapsed: 1.02 seconds
    # Memory usage: 388.7 MB
