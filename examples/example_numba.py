from pathlib import Path
import numpy as np
import vectorbtpro as vbt

from vectorbtpro_templates import *

try:
    DATA_DIR = Path(__file__).resolve().parent
except:
    pass


if __name__ == "__main__":

    # Load historical data from CSV
    path = DATA_DIR / "csv" / "NQ=F_ohlcv_data.csv"
    data = get_data_from_csv(path, sep=";")
    close = vbt.to_1d_array(data.close)
    
    # Get the annualization factor to calculate sharpe using ReturnsAccessor.get_ann_factor
    # Source: https://vectorbt.pro/pvt_1606a55a/api/returns/accessors/#vectorbtpro.returns.accessors.ReturnsAccessor.get_ann_factor
    ann_factor = int(vbt.pd_acc.returns.get_ann_factor(freq='D'))
    
    # Run Custom Indicator
    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        st_nb = StrategyNumba.run(
            data.close,
            param_product=True,
            execute_kwargs=dict(n_chunks="auto", distribute="chunks", engine="pathos"),
            **default_params
        )
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())

    # Available outputs:
    print(st_nb.entries)
    print(st_nb.exits)
    
    # Single parameter
    # ---------------
    
    # Run signals nb
    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        entries, exits = get_signals_nb(close, **default_single_params)
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())
    
    # Run portfolio nb
    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        sim_out = get_portfolio_nb(close, entries, exits)
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())
    
    # Run metrics nb
    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        sharpe_metrics = get_metrics_nb(sim_out, ann_factor)
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())
    
    # Run pipeline nb
    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        sharpe_pipeline = pipeline_nb(close, ann_factor=ann_factor, **default_single_params)
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())

    # Check outputs
    pf = vbt.Portfolio(data.symbol_wrapper.regroup(group_by=True), sim_out, close=data.close)
    sharpe_portfolio = pf.sharpe_ratio
    assert sharpe_metrics == sharpe_pipeline == sharpe_portfolio
    
    # Multiple parameters
    # -------------------
    
    # Run @Parametrized [automatic parameter combinations]
    pipe = vbt.parameterized(
        pipeline_nb,
        merge_func="concat",
        # -> Processes: engine="pathos", distribute="chunks", n_chunks="auto"
        # -> Threads: engine="threadpool" / "dask", chunk_len="auto"
        engine="threadpool", chunk_len="auto"
    )
    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        sharpes_parametrized = pipe(close, ann_factor=ann_factor, **default_vbt_params)
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())
    
    # Note:
    # Instead of testing the full parameter grid, we can adopt a statistical approach.
    # Control number of random trials with `random_subset`
    # Doc: https://vectorbt.pro/pvt_1606a55a/api/records/base/#saving-and-loading
    
    # Construct the parameter grid manually
    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        param_product, param_index = vbt.combine_params(default_vbt_params)
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())
    
    # Total number of parameter combinations
    n_params = len(param_index) # <- Number of parameter combinations to process during each iteration of the while-loop
 
    # Run @chunked pipeline nb
    
    # The parameterized decorator has two major limitations though: it runs
    # only one parameter combination at a time, and it needs to build the
    # parameter grid fully, even when querying a random subset.
    
    # Test wrapper function
    
    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        sharpes_chunked_wrapper = chunked_wrapper_nb(
            n_params, 
            close, 
            fastperiod=param_product["fastperiod"],
            slowperiod=param_product["slowperiod"],
            signalperiod=param_product["signalperiod"],
            timeperiod=param_product["timeperiod"],
            window=param_product["window"],
            alpha=param_product["alpha"],
            ann_factor=ann_factor,
            _chunk_len='auto',
            _execute_kwargs=dict(chunk_len="auto", engine="threadpool")
            # Any argument passed to the chunked decorator can be overridden
            # during the runtime using the same argument but prefixed with
            # an underscore _
        )
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())
    
    # Get the output
    print(pd.Series(sharpes_chunked_wrapper, index=param_index))
    
    # Test pipeline
    
    with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
        sharpes_chunked_pipeline = pipeline_chunked_nb(
            close,
            default_vbt_params,
            ann_factor=ann_factor,
            path=None,
            to_pd_series=False,
            _chunk_len='auto',
            _execute_kwargs=dict(chunk_len="auto", engine="threadpool")
        )
    print('Time elapsed:', timer.elapsed())
    print('Memory usage:', tracer.peak_usage())

 
    # Check outputs
    np.testing.assert_array_equal(sharpes_parametrized.values, sharpes_chunked_wrapper)
    np.testing.assert_array_equal(sharpes_chunked_pipeline, sharpes_chunked_pipeline)



    
 
