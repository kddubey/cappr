from __future__ import annotations
from typing import Optional, Sequence

import numpy as np


def prior(prior: Optional[Sequence[float]] = None):
    """
    Raises an error if `prior` is not `None`, or if it's not a 1-D `Sequence` which sums
    to 1.
    """
    if prior is None:  ## it's a uniform prior, no need to check anything
        return
    if not isinstance(prior, (Sequence, np.ndarray)):
        raise TypeError("prior must be None or a sequence.")
    if len(np.shape(prior)) != 1:
        raise ValueError("prior must be 1-D.")
    prior_arr = np.array(prior, dtype=float)  ## try casting to float
    if not np.isclose(prior_arr.sum(), 1):
        raise ValueError("prior must sum to 1.")
