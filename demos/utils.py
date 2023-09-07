from __future__ import annotations
from typing import Optional, Union

from IPython.display import display
import pandas as pd


def display_df(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    num_rows: Union[int, None] = 3,
):
    """
    Displays `df.head(num_rows)[columns]` without truncating columns. If
    possible, render any newlines.
    """
    if columns is None:
        columns = df.columns
    if num_rows is None:
        num_rows = len(df)
    df_head_styled = df.head(num_rows)[columns].style
    with pd.option_context("max_colwidth", None):
        ## I'm not sure why try-except doesn't work w/ display(), so instead
        ## check the necessary uniqueness condition before running it
        if df.index.is_unique:
            display(
                df_head_styled.set_properties(
                    **{"text-align": "left", "white-space": "pre-wrap"}
                )
            )
        else:
            ## `Styler.apply` and `.applymap` are not compatible with non-unique
            ## index or columns
            display(df_head_styled)


def remove_suffix(string: str, suffix: str):
    if string.endswith(suffix):
        return string[: -len(suffix)]
    return string


def remove_prefix(string: str, prefix: str) -> str:
    if string.startswith(prefix):
        return string[len(prefix) :]
    return string
