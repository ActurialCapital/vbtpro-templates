from pathlib import Path
import vectorbtpro as vbt

from vectorbtpro_templates import (
    get_data_from_csv,
    chunked_wrapper_nb,
    ParamTemplate,
    np_list_arange
)

try:
    DATA_DIR = Path(__file__).resolve().parent
except:
    pass

# Build custom parameter template
param_template = ParamTemplate(
    fastperiod=range(5, 15),
    slowperiod=range(20, 40),
    signalperiod=range(3, 13),
    timeperiod=range(2, 32),
    window=range(5, 20),
    alpha=np_list_arange(0.5, 3.6, 0.2)
)

# Adds vbt.Param to each parameter
all_vbt_params = {
    key: vbt.Param(value) for key, value in param_template._asdict().items()
}


if __name__ == "__main__":

    # Data
    path = DATA_DIR / "csv" / "NQ=F_ohlcv_data.csv"
    data = get_data_from_csv(path, sep=";")
    close = vbt.to_1d_array(data.close) # np.array 

    # Save simulation on disk
    # Delete previously generater directory (if any)
    vbt.remove_dir("temp", with_contents=True, missing_ok=True)
    # Create temporary directory
    vbt.make_dir("temp")
    # File name
    FILE_NAME = "temp/sharpes.pickle"
    
    # Get the annualization factor to calculate sharpe using ReturnsAccessor.get_ann_factor
    # Source: https://vectorbt.pro/pvt_1606a55a/api/returns/accessors/#vectorbtpro.returns.accessors.ReturnsAccessor.get_ann_factor
    ann_factor = int(vbt.pd_acc.returns.get_ann_factor(freq='D'))
    
    print('[INFO] Starting parameter combination process...')

    with vbt.Timer() as timer:
        param_product, param_index = vbt.combine_params(all_vbt_params)

    # Total number of parameter combinations
    n_params = len(param_index)

    print('[INFO] Time elapsed for combine_params:', timer.elapsed())
    print(f"[INFO] Total number of parameter combinations: {n_params:,d}")
    print(f"[INFO] Chunk size: {100_000:,d}")
    print('[INFO] Number of chunks [CPU cores=8]:', n_params / 100_000 / 8)
    print('[INFO] Processing chunks...')

    with vbt.Timer() as timer, vbt.MemTracer() as tracer:
        
        if not vbt.file_exists(FILE_NAME):
            sharpes = chunked_wrapper_nb(
                n_params, 
                close, 
                fastperiod=param_product["fastperiod"],
                slowperiod=param_product["slowperiod"],
                signalperiod=param_product["signalperiod"],
                timeperiod=param_product["timeperiod"],
                window=param_product["window"],
                alpha=param_product["alpha"],
                ann_factor=ann_factor,
                _chunk_len=100_000, # Pass at most n parameter combinations at a time
                # Apart from chunking the parameter arrays, we can also put chunks themselves into so-called "super chunks". 
                # Each super chunk will consist of as many chunks as there are CPU cores - one per thread.
                _execute_kwargs=dict(chunk_len='auto', engine="threadpool") 
                # Any argument passed to the chunked decorator can be overridden
                # during the runtime using the same argument but prefixed with
                # an underscore _
            )
            vbt.save(sharpes, FILE_NAME)
        else:
            sharpes = vbt.load(FILE_NAME)
    
    print('[INFO] Overall Progress:')
    print('[INFO] Time elapsed:', timer.elapsed())
    print('[INFO] Memory usage:', tracer.peak_usage())
    print(sharpes)
