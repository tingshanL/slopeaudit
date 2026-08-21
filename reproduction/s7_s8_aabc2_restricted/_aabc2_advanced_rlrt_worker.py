#!/usr/bin/env python3
"""Process-worker state for the advanced AABC2 RLRT driver."""
from types import SimpleNamespace

import numpy as np

import _aabc2_advanced_rlrt_core as core


def initialize(outcomes, analysis_data, observed_null_state):
    """Initialize one spawned worker with plain, pickle-safe null-fit state."""
    core.configure_analysis(outcomes, analysis_data)
    panels = {}

    for outcome, state in observed_null_state.items():
        if state is None:
            result = None
        else:
            result = SimpleNamespace(
                model=SimpleNamespace(
                    exog=np.asarray(state["exog"], float),
                ),
                fe_params=np.asarray(state["fe_params"], float),
                cov_re=np.asarray(state["cov_re"], float),
                scale=float(state["scale"]),
            )
        panels[(outcome, "null")] = {"result": result}

    core.OBSERVED_PANELS = panels


def simulate_pair(task):
    outcome, replicate = task
    return core.simulate_one_null_rlrt(outcome, replicate)


def repair_ventricular(replicate):
    return core.repair_one_ventricular_rlrt(replicate)
