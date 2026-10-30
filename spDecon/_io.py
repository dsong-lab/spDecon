"""CSV identifiers and JSON output shared by the runnable workflows."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_frame(path, *, text=False):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        header = next(csv.reader(stream), [])
    if len(header) < 2 or len(set(header)) != len(header) or any(not h for h in header[1:]):
        raise ValueError(f'CSV needs an ID column and unique nonempty column names: {path}')
    # Setting the index after parsing prevents pandas from inferring numeric IDs.
    options = dict(dtype=str) if text else dict(converters={0: str})
    frame = pd.read_csv(path, keep_default_na=False, **options)
    frame.index = pd.Index(frame.pop(frame.columns[0]), name=header[0] or None)
    if not frame.index.is_unique or (frame.index == '').any():
        raise ValueError(f'Duplicate or empty identifiers: {path}')
    return frame


def write_json(path, value):
    def serial(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=serial)+'\n')
