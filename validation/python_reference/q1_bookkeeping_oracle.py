#!/usr/bin/env python3
"""Independent Question-1 bookkeeping oracle.

This script does not fit mixed models. It independently checks the frozen
drop-vs-zero denominator and plus-one p-value rules using a tiny hand-specified
fixture. It is meant to arbitrate bookkeeping disagreements, not lme4 fitting.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "q1_bookkeeping_fixture.json"


def plus_one(k: int, n: int) -> float:
    if not (0 <= k <= n):
        raise ValueError((k, n))
    return (k + 1) / (n + 1)


def main() -> None:
    obj = json.loads(FIXTURE.read_text(encoding="utf-8"))
    reps = obj["replicates"]
    observed = float(obj["observed_LR"])

    drop = [r for r in reps if r["status"] in {"positive", "zero"}]
    zero = [
        r for r in reps
        if r["status"] in {"positive", "zero", "small_negative"}
    ]

    k_drop = sum(float(r["LR_raw"]) >= observed for r in drop)
    k_zero = sum(float(r["LR_zero"]) >= observed for r in zero)

    got = {
        "denominator_drop": len(drop),
        "denominator_zero": len(zero),
        "exceedances_drop": k_drop,
        "exceedances_zero": k_zero,
        "p_drop": plus_one(k_drop, len(drop)),
        "p_zero": plus_one(k_zero, len(zero)),
        "n_small_negative": sum(r["status"] == "small_negative" for r in reps),
        "n_material_negative": sum(r["status"] == "material_negative" for r in reps),
        "n_failure": sum(r["status"] == "failure" for r in reps),
    }

    expected = obj["expected"]

    for key, value in expected.items():
        if isinstance(value, float):
            if abs(float(got[key]) - value) > 1e-15:
                raise AssertionError((key, got[key], value))
        elif got[key] != value:
            raise AssertionError((key, got[key], value))

    print(json.dumps(got, indent=2))
    print("Question-1 bookkeeping oracle: PASS")


if __name__ == "__main__":
    main()
