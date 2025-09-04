from __future__ import annotations

from pathlib import Path
from typing import Dict

from HH4b.hh_vars import samples_run3
from HH4b.postprocessing.postprocessing import load_run3_samples

# TODO: REVIEW

def load_year(
    data_root: str | Path,
    year: str,
    *,
    txbb_version: str,
    bdt_version: str,
    load_systematics: bool = False,
    reorder_txbb: bool = True,
    mass_str: str = "bbFatJetParTmassVis",
) -> Dict[str, "pd.DataFrame"]:
    """
    Load a single year's samples using HH4b's loader, returning an events_dict.
    Normalization to finalWeight is applied by HH4b's utils.
    """
    events = load_run3_samples(
        input_dir=str(data_root),
        year=year,
        samples_run3=samples_run3,
        reorder_txbb=reorder_txbb,
        load_systematics=load_systematics,
        txbb_version=txbb_version,
        scale_and_smear=False,
        mass_str=mass_str,
        bdt_version=bdt_version,
    )
    return events
