"""Step 1: Curate clips from Something-Something V2.

Reads SSv2 train.json + validation.json, filters via configs/verb_mapping.yaml,
and writes a list of curated clips with their mapped verb.

If --balance is on (default), keeps at most `target_per_verb` clips per verb,
preferring SSv2 train split over validation, and within a split picks at
random with a fixed seed for reproducibility.

Output: outputs/curated_clips.json with shape:
    {
        "n": <total>,
        "per_verb": {open: 125, close: 125, ...},
        "clips": [SSv2Clip.to_dict(), ...],
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ssv2_loader import collect_ssv2_clips


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--verb_mapping", default="configs/verb_mapping.yaml")
    ap.add_argument("--out", default="outputs/curated_clips.json")
    ap.add_argument("--no_balance", action="store_true",
                    help="Keep ALL mapped clips instead of subsampling to target_per_verb.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_yaml(Path(args.config))
    vm = _load_yaml(Path(args.verb_mapping))["verb_mapping"]

    if not cfg["source"]["ssv2"]["enabled"]:
        raise SystemExit("source.ssv2.enabled is false in config; nothing to do.")

    ssv2_root = Path(cfg["source"]["ssv2"]["root"])
    splits_used = cfg["source"]["ssv2"]["splits_used"]

    clips = collect_ssv2_clips(
        ssv2_root=ssv2_root,
        verb_mapping=vm,
        splits_used=splits_used,
        videos_subdir=cfg["source"]["ssv2"]["videos_dir"],
        labels_subdir=cfg["source"]["ssv2"]["labels_dir"],
        verbose=True,
    )
    if not clips:
        raise SystemExit(
            "No clips matched verb_mapping.yaml. Check the templates are exactly "
            "what SSv2's labels.json uses (case + brackets matter)."
        )

    # ----- balance per verb -----
    rng = random.Random(cfg["runtime"]["seed"])
    by_verb = defaultdict(list)
    for c in clips:
        by_verb[c.our_verb].append(c)

    target = cfg["target_per_verb"]
    min_target = cfg["min_target_per_verb"]

    kept = []
    if args.no_balance:
        kept = list(clips)
        logging.info("--no_balance: keeping ALL %d mapped clips", len(kept))
    else:
        for verb in sorted(by_verb.keys()):
            pool = by_verb[verb]
            # Prefer train > validation (train tends to be more diverse)
            pool.sort(key=lambda c: (0 if c.source_split == "train" else 1))
            if len(pool) <= target:
                kept.extend(pool)
                logging.info("verb=%s : kept all %d (under target %d)",
                             verb, len(pool), target)
                if len(pool) < min_target:
                    logging.warning(
                        "verb=%s under min_target_per_verb=%d, only %d available",
                        verb, min_target, len(pool),
                    )
            else:
                # Take first up to (target/2) from train, rest from validation,
                # then fill with random sampling within each pool.
                rng.shuffle(pool)
                kept.extend(pool[:target])
                logging.info("verb=%s : sampled %d / %d available",
                             verb, target, len(pool))

    # ----- per-verb counter for the output header -----
    per_verb = defaultdict(int)
    for c in kept:
        per_verb[c.our_verb] += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n":         len(kept),
        "per_verb":  dict(sorted(per_verb.items())),
        "clips":     [c.to_dict() for c in kept],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    logging.info("Wrote %d clips -> %s", len(kept), out_path)
    logging.info("Per-verb breakdown:")
    for v, n in sorted(per_verb.items()):
        logging.info("  %-10s %d", v, n)


if __name__ == "__main__":
    main()
