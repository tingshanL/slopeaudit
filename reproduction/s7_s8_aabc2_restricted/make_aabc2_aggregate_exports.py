#!/usr/bin/env python3
"""Inventory AABC2 source artifacts through a conservative router.

This router operates inside an authorized AABC2 analysis environment. It maps
internal filenames to semantic names, records hashes and source classifications,
and prevents automatic copying of restricted artifacts. Manifest mode copies no
AABC2-derived source file; approved-copy mode requires explicit approval.

The router policy describes direct source artifacts. The reviewed 29-file,
identifier-free aggregate result set under `results/frozen_summaries/aabc2/`
has a separate manifest and privacy review. Participant-
linked outputs classified as restricted_never_export remain uncopiable in
every router mode.

No absolute source path is written to the router manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict


APPROVAL_PHRASE = "AABC2 DERIVATIVE EXPORT APPROVED"
POLICY_PATH = Path(__file__).with_name("aabc2_release_output_policy.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-mapping", required=True, type=Path, help="Private provenance mapping from semantic artifact IDs to local frozen filenames.")
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--fit-dir", required=True, type=Path)
    parser.add_argument("--reliability-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-dir", required=True, type=Path)
    parser.add_argument("--stability-dir", required=True, type=Path)
    parser.add_argument("--advanced-dir", required=True, type=Path)
    parser.add_argument("--legacy-v3-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("manifest", "approved-copy"),
        default="manifest",
        help=(
            "manifest inventories files but copies no AABC2-derived results; "
            "approved-copy requires explicit written-approval acknowledgement"
        ),
    )
    parser.add_argument(
        "--approval-file",
        type=Path,
        help=(
            "Required only in approved-copy mode.  The file must contain the "
            f"exact phrase: {APPROVAL_PHRASE}"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_fresh_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"Output directory must be fresh/empty: {path}")
    else:
        path.mkdir(parents=True)
    return path


def load_policy() -> dict:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    entries = policy.get("entries", [])
    if not entries:
        raise ValueError("Release output policy contains no entries")

    allowed = {
        "hold_pending_written_approval",
        "restricted_never_export",
    }
    classifications = {row["classification"] for row in entries}
    unexpected = classifications - allowed
    if unexpected:
        raise ValueError(f"Unexpected release classifications: {sorted(unexpected)}")

    release_names = [row["release_filename"] for row in entries]
    if len(release_names) != len(set(release_names)):
        raise ValueError("Release output policy contains duplicate semantic filenames")
    if any(Path(name).name != name for name in release_names):
        raise ValueError("release_filename must be a basename, not a path")
    if any(name[:1].isdigit() for name in release_names):
        raise ValueError("Semantic release filenames may not start with processing-order numbers")

    return policy


def approval_is_valid(path: Path | None) -> bool:
    if path is None:
        return False
    text = path.expanduser().read_text(encoding="utf-8")
    return APPROVAL_PHRASE in text


def main() -> None:
    args = parse_args()
    output_dir = require_fresh_output(args.output_dir)
    policy = load_policy()
    source_mapping = json.loads(
        args.source_mapping.expanduser().read_text(encoding="utf-8")
    )
    mapping_rows = source_mapping.get("entries", [])
    mapping_by_id = {row["artifact_id"]: row for row in mapping_rows}
    if len(mapping_by_id) != len(mapping_rows):
        raise ValueError("Source mapping contains duplicate artifact IDs")
    policy_ids = {row["artifact_id"] for row in policy["entries"]}
    if set(mapping_by_id) != policy_ids:
        missing = sorted(policy_ids - set(mapping_by_id))
        extra = sorted(set(mapping_by_id) - policy_ids)
        raise ValueError(
            f"Source mapping does not match release policy; missing={missing}, extra={extra}"
        )

    roots: Dict[str, Path] = {
        "prepared": args.prepared_dir.expanduser().resolve(),
        "fit": args.fit_dir.expanduser().resolve(),
        "reliability": args.reliability_dir.expanduser().resolve(),
        "bootstrap": args.bootstrap_dir.expanduser().resolve(),
        "stability": args.stability_dir.expanduser().resolve(),
        "advanced": args.advanced_dir.expanduser().resolve(),
        "legacy_v3": args.legacy_v3_dir.expanduser().resolve(),
    }

    unknown_groups = {
        row["source_group"] for row in mapping_rows
    } - set(roots)
    if unknown_groups:
        raise ValueError(f"Policy uses unknown source groups: {sorted(unknown_groups)}")

    if args.mode == "approved-copy" and not approval_is_valid(args.approval_file):
        raise PermissionError(
            "approved-copy mode requires an approval file containing the exact "
            f"phrase: {APPROVAL_PHRASE}"
        )

    rows = []
    copy_dir = output_dir / "approved"
    copied = 0

    for entry in policy["entries"]:
        source_entry = mapping_by_id[entry["artifact_id"]]
        group = source_entry["source_group"]
        relative = Path(source_entry["source_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source_relative_path: {relative}")

        source = roots[group] / relative
        exists = source.is_file()
        classification = entry["classification"]

        action = "not_found"
        if exists:
            if classification == "restricted_never_export":
                action = "restricted_never_export"
            elif args.mode == "manifest":
                action = "hold_pending_written_approval"
            else:
                copy_dir.mkdir(parents=True, exist_ok=True)
                destination = copy_dir / entry["release_filename"]
                if destination.exists():
                    raise FileExistsError(f"Refusing to overwrite: {destination}")
                shutil.copyfile(source, destination)
                if sha256(source) != sha256(destination):
                    raise RuntimeError(f"Copy hash mismatch: {entry['release_filename']}")
                action = "copied_after_approval"
                copied += 1

        rows.append(
            {
                "artifact_id": entry["artifact_id"],
                "source_group": group,
                "source_relative_path": source_entry["source_relative_path"],
                "release_filename": entry["release_filename"],
                "classification": classification,
                "source_exists": bool(exists),
                "bytes": source.stat().st_size if exists else "",
                "sha256": sha256(source) if exists else "",
                "action": action,
            }
        )

    manifest = output_dir / "aabc2_export_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    restricted_copied = [
        row for row in rows
        if row["classification"] == "restricted_never_export"
        and row["action"] == "copied_after_approval"
    ]
    if restricted_copied:
        raise RuntimeError("Restricted-never-export file was copied")

    print("AABC2 release export routing complete.")
    print("mode:", args.mode)
    print("policy entries:", len(rows))
    print("existing source files:", sum(bool(row["source_exists"]) for row in rows))
    print("copied result files:", copied)
    print("restricted-never-export copied:", 0)
    print("manifest:", manifest)


if __name__ == "__main__":
    main()
