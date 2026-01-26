import os
import pandas as pd
from src.data_loader import load_csv


def test_load_csv_file_not_found(tmp_path):
    p = tmp_path / "nope.csv"
    try:
        load_csv(p)
    except FileNotFoundError:
        assert True
    else:
        assert False, "Expected FileNotFoundError for missing file"


def test_load_csv_reads(tmp_path):
    p = tmp_path / "sample.csv"
    df = pd.DataFrame({'a':[1,2,3]})
    df.to_csv(p, index=False)
    loaded = load_csv(p)
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape == (3,1)
