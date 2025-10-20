import os
from typing import Iterable


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def batched(iterable: Iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


