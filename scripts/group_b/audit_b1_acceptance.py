"""Independent acceptance check for Group B B1 audit outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    monthly = pd.read_csv(args.monthly)

    errors: list[str] = []
    if audit.get("status") != "PASS":
        errors.append("audit status is not PASS")
    if audit.get("analysis_status") != "FROZEN_BEFORE_MODELING":
        errors.append("analysis plan was not frozen before modeling")
    if config.get("analysis_id") != audit.get("analysis_id"):
        errors.append("analysis ID mismatch")
    if audit.get("sample_file_count") != 48:
        errors.append("sample file count is not 48")
    if audit.get("observed_month_count") != 48 or len(monthly) != 48:
        errors.append("monthly coverage is not 48 months")
    if tuple(monthly.loc[0, ["year", "month"]].astype(int)) != (2022, 1):
        errors.append("first month is not 2022-01")
    if tuple(monthly.loc[len(monthly) - 1, ["year", "month"]].astype(int)) != (2025, 12):
        errors.append("last month is not 2025-12")
    if int(monthly["group_b_rows"].sum()) != int(audit.get("group_b_rows", -1)):
        errors.append("monthly Group B rows do not reconcile")
    if any(audit.get("missing_counts", {}).values()):
        errors.append("required missing counts are not zero")
    if any(audit.get("nonfinite_counts", {}).values()):
        errors.append("nonfinite counts are not zero")
    if any(audit.get("invalid_counts", {}).values()):
        errors.append("invalid population counts are not zero")

    tip_quantiles = audit.get("quantiles", {}).get("tips", {})
    ordered = [tip_quantiles.get(key) for key in ("0.0", "0.5", "0.99", "1.0")]
    if any(value is None for value in ordered) or ordered != sorted(ordered):
        errors.append("tip quantiles are missing or unordered")

    if errors:
        print("GROUP B B1 ACCEPTANCE: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"group_b_rows={audit['group_b_rows']}")
    print(f"sample_file_count={audit['sample_file_count']}")
    print(f"observed_month_count={audit['observed_month_count']}")
    print(f"tip_median={tip_quantiles['0.5']}")
    print(f"tip_p99={tip_quantiles['0.99']}")
    print(f"tip_p999={tip_quantiles['0.999']}")
    print("GROUP B B1 ACCEPTANCE: PASS")


if __name__ == "__main__":
    main()
