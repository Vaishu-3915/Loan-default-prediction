from typing import Iterator, Optional, List
import pandas as pd
from config import DataConfig


def read_csv_in_chunks(cfg: DataConfig, usecols: Optional[List[str]] = None, dtypes: Optional[dict] = None) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(
        cfg.input_csv_path,
        chunksize=cfg.chunk_size,
        low_memory=cfg.low_memory,
        usecols=usecols,
        dtype=dtypes,
    )
    for chunk in reader:
        yield chunk


