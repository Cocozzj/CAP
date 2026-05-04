"""One-shot rename: standardize camera filenames across all rendered
trajectories to `cam0 / cam1 / cam2`.

Background:
    The adaptive camera planner used for articulated trajectories writes
    files as `cam0.mp4 / cam1.mp4 / cam2.mp4` (and the corresponding keys
    in cameras.json). The pyrender-based soft renderer follows the static
    camera names defined in configs/cameras.yaml, which are
    `front / side / high_oblique`. This inconsistency hurts downstream
    code (Step 4 / Step 5 / dataloaders), so we normalize everyone to
    cam0/cam1/cam2.

What this script does, per trajectory folder:
    - rename mp4 files (front.mp4 -> cam0.mp4, side.mp4 -> cam1.mp4,
      high_oblique.mp4 -> cam2.mp4)
    - update cameras.json: rename keys, write back

It is idempotent — running twice is safe (already-cam{0,1,2} folders are
left alone).

Usage:
    python scripts/normalize_camera_names.py --data_dir outputs/data
    python scripts/normalize_camera_names.py --data_dir outputs/data --dry_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Soft-renderer convention -> articulated convention
RENAME_MAP = {
    "front": "cam0",
    "side": "cam1",
    "high_oblique": "cam2",
}


def normalize_one(traj_dir: Path, dry_run: bool = False) -> str:
    """Returns one of: 'already_normalized', 'renamed', 'partial_renamed',
    'no_cameras_json', 'unknown_scheme'."""
    cams_path = traj_dir / "cameras.json"
    if not cams_path.exists():
        return "no_cameras_json"

    with open(cams_path) as f:
        cams = json.load(f)

    keys = set(cams.keys())
    target_keys = {"cam0", "cam1", "cam2"}
    soft_keys = set(RENAME_MAP.keys())

    if keys == target_keys:
        return "already_normalized"
    if keys != soft_keys:
        # Some other unexpected scheme; don't touch
        return f"unknown_scheme: {sorted(keys)}"

    # Plan: rename mp4 files + update cameras.json
    renames = []
    for old_name, new_name in RENAME_MAP.items():
        old_mp4 = traj_dir / f"{old_name}.mp4"
        new_mp4 = traj_dir / f"{new_name}.mp4"
        if not old_mp4.exists():
            return f"missing_mp4: {old_mp4.name}"
        if new_mp4.exists():
            return f"target_already_exists: {new_mp4.name}"
        renames.append((old_mp4, new_mp4))

    if dry_run:
        return "would_rename"

    # Execute renames atomically per file
    for old_mp4, new_mp4 in renames:
        old_mp4.rename(new_mp4)

    # Build new cameras.json with renamed keys, preserving original order
    new_cams = {}
    for old_key in cams:
        new_cams[RENAME_MAP[old_key]] = cams[old_key]
    with open(cams_path, "w") as f:
        json.dump(new_cams, f, indent=2)

    return "renamed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="outputs/data")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print plan without making changes")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    traj_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Scanning {len(traj_dirs)} trajectory folders in {data_dir}")

    counts = {
        "already_normalized": 0,
        "renamed": 0,
        "would_rename": 0,
        "no_cameras_json": 0,
        "errors": 0,
    }
    error_samples = []
    for tdir in traj_dirs:
        result = normalize_one(tdir, dry_run=args.dry_run)
        if result in counts:
            counts[result] += 1
        else:
            counts["errors"] += 1
            if len(error_samples) < 10:
                error_samples.append(f"{tdir.name}: {result}")

    print("\nSummary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if error_samples:
        print("\nFirst few error/skip cases:")
        for s in error_samples:
            print(f"  {s}")

    if args.dry_run:
        print("\n[dry_run] No changes written.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
