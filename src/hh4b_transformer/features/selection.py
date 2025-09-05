from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from HH4b.utils import make_selection

# TODO: REVIEW
# no selections applied in paper, this may not be necessary


def apply_selection(
    events_dict: Dict[str, pd.DataFrame],
    var_cuts: Dict[str, list],
    *,
    weight_key: str = "finalWeight",
    jshift: str = "",
) -> Tuple[Dict[str, "np.ndarray"], "pd.DataFrame"]:
    return make_selection(
        var_cuts, events_dict, weight_key=weight_key, prev_cutflow=None, jshift=jshift
    )
