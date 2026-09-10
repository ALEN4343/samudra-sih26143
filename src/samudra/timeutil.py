"""Timestamp conversion. All timestamps are UTC.

Exists because of a real bug: pandas 3.0 stores datetimes as datetime64[us] by
default, not [ns]. The widespread idiom

    series.astype("int64") / 1e9

therefore silently returns microseconds divided by 1e9 — timestamps 1000x too
small — and every downstream time comparison quietly matches nothing instead of
raising. Convert through this module, never by hand.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch_seconds(s: pd.Series) -> np.ndarray:
    """Datetime series -> float epoch seconds UTC, whatever the storage unit."""
    if not pd.api.types.is_datetime64_any_dtype(s):
        raise TypeError(f"expected a datetime series, got dtype {s.dtype}")
    s = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    return ((s - _EPOCH) / pd.Timedelta("1s")).to_numpy(dtype=float)


def to_utc(t: datetime) -> datetime:
    """Attach UTC to a naive datetime, convert an aware one."""
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def from_epoch(seconds: float) -> datetime:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc)
