"""Input DataFrame contract."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("cell_type",)

# Used if present, silently skipped if absent.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "cell_type_class",
    "cell_type_description",
    "assembly",
    "title",
    "antigen",
    "tf_name",
    "antigen_class",
    "experiment_id",
)


def validate_input(df: pd.DataFrame) -> None:
    """Raise ValueError if the input DataFrame is missing required columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}. "
            f"Expected at least {list(REQUIRED_COLUMNS)}."
        )
