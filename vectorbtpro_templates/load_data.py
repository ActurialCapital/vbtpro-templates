from pathlib import Path
import pandas as pd
import vectorbtpro as vbt
import vectorbtpro._typing as tp  # -> vbt typing extension


def get_data_from_csv(
    path: Path, 
    date: tp.Optional[str] = 'Date',
    **read_excel_kwargs
) -> vbt.Data:
    """
    Load data from a CSV file and convert it to a `vbt.Data` object for compatibility 
    with vectorbt.pro. It parses the date column as the index for time-series analysis.
    """
    df = pd.read_csv(path, dayfirst=True, parse_dates=[date], index_col=date, dtype=float, **read_excel_kwargs)
    # Better to work directly with vbt.Data object
    # https://vectorbt.pro/pvt_12537e02/documentation/data/
    return vbt.Data.from_data(df)
