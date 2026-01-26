import pandas as pd
from pathlib import Path


def load_csv(path, nrows=None):
    """Load a CSV into a DataFrame.

    Args:
        path (str | Path): path to CSV file
        nrows (int, optional): read only N rows for quick tests

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError: if path doesn't exist
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p, nrows=nrows)
