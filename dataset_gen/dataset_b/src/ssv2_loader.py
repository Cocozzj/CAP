"""Read Something-Something V2 annotations and return per-clip records.

SSv2's annotation format (as released by Qualcomm / 20BN):
    labels/
        labels.json       # {"<class_name>": "<class_index_str>", ...}
        train.json        # [{"id": "1234", "label": "<class_name>",
                          #   "template": "<class_name_with_brackets>",
                          #   "placeholders": ["something1", "something2"]}, ...]
        validation.json   # same shape

Videos are at:
    videos/<id>.webm

Some redistributions use .mp4 instead of .webm; we accept either.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SSv2Clip:
    """One clip after parsing SSv2 annotations + verb mapping."""
    clip_id: str                 # SSv2 numeric id, e.g. "1234"
    source_split: str            # 'train' | 'validation'
    raw_label: str               # original SSv2 class name (with [something] placeholders filled)
    template: str                # original template with [something] etc.
    placeholders: List[str]      # what nouns SSv2 says fill the [something] slots
    our_verb: str                # mapped to one of: open / close / pull / push / rotate / squeeze / fold / pour
    video_path: Path             # absolute path to .webm or .mp4

    def to_dict(self) -> dict:
        return {
            "clip_id":      self.clip_id,
            "source_split": self.source_split,
            "raw_label":    self.raw_label,
            "template":     self.template,
            "placeholders": self.placeholders,
            "our_verb":     self.our_verb,
            "video_path":   str(self.video_path),
        }


def load_labels_index(labels_dir: Path) -> Dict[str, str]:
    """Returns dict mapping class_name -> class_index_str (from labels.json)."""
    p = labels_dir / "labels.json"
    if not p.exists():
        # Some releases name it 'something-something-v2-labels.json'
        candidates = list(labels_dir.glob("*labels*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"labels.json not found in {labels_dir}. "
                f"Expected files like labels.json or something-something-v2-labels.json"
            )
        p = candidates[0]
    with open(p) as f:
        data = json.load(f)
    # data may be {class_name: index_str} or list of {"name": ..., "id": ...}
    if isinstance(data, dict):
        return data
    return {entry["name"]: str(entry.get("id", i)) for i, entry in enumerate(data)}


def load_split_annotations(labels_dir: Path, split: str) -> List[dict]:
    """Returns list of clip annotation dicts for a given split."""
    candidates = [
        labels_dir / f"{split}.json",
        labels_dir / f"something-something-v2-{split}.json",
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise FileNotFoundError(
            f"{split}.json not found in {labels_dir}. Tried: "
            + ", ".join(str(c) for c in candidates)
        )
    with open(p) as f:
        return json.load(f)


def find_video_file(videos_dir: Path, clip_id: str) -> Optional[Path]:
    """Return the path to clip's video, trying .webm then .mp4."""
    for ext in (".webm", ".mp4", ".avi"):
        p = videos_dir / f"{clip_id}{ext}"
        if p.exists():
            return p
    return None


def _normalize_label(label: str) -> str:
    """Normalize a class label so we can compare to verb_mapping.yaml entries.

    SSv2 labels in train.json are filled in with the user's nouns (e.g.
    'Opening box'), while labels.json keeps the bracketed template
    'Opening [something]'. We compare against the template form so the
    mapping doesn't depend on what noun was used.
    """
    return label.strip()


def _build_template_from_annotation(ann: dict) -> str:
    """Try to recover the bracketed template from a train/validation entry.

    Each annotation may carry a 'template' field (preferred). If not, we
    can't recover the canonical form; fall back to the filled label.
    """
    if "template" in ann:
        return _normalize_label(ann["template"])
    return _normalize_label(ann.get("label", ""))


def collect_ssv2_clips(
    ssv2_root: Path,
    verb_mapping: Dict[str, List[str]],
    splits_used: Sequence[str] = ("train", "validation"),
    videos_subdir: str = "videos",
    labels_subdir: str = "labels",
    *,
    verbose: bool = True,
) -> List[SSv2Clip]:
    """Walk the requested SSv2 splits and return every clip whose template
    appears in `verb_mapping`. Verb mapping is dict {our_verb: [list of templates]}.
    """
    ssv2_root = Path(ssv2_root)
    videos_dir = ssv2_root / videos_subdir
    labels_dir = ssv2_root / labels_subdir
    if not videos_dir.exists():
        raise FileNotFoundError(f"SSv2 videos dir not found: {videos_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"SSv2 labels dir not found: {labels_dir}")

    # Inverse mapping: template -> our_verb (case-sensitive exact match)
    template_to_verb: Dict[str, str] = {}
    for our_verb, templates in verb_mapping.items():
        for t in templates:
            template_to_verb[_normalize_label(t)] = our_verb

    if verbose:
        logger.info("verb_mapping loaded: %d templates across %d verbs",
                    len(template_to_verb), len(verb_mapping))

    out: List[SSv2Clip] = []
    seen_unmapped: Dict[str, int] = defaultdict(int)
    n_missing_video = 0

    for split in splits_used:
        anns = load_split_annotations(labels_dir, split)
        if verbose:
            logger.info("Loaded %d annotations from %s", len(anns), split)
        for ann in anns:
            template = _build_template_from_annotation(ann)
            verb = template_to_verb.get(template)
            if verb is None:
                seen_unmapped[template] += 1
                continue
            clip_id = str(ann["id"])
            vp = find_video_file(videos_dir, clip_id)
            if vp is None:
                n_missing_video += 1
                continue
            out.append(SSv2Clip(
                clip_id=clip_id,
                source_split=split,
                raw_label=ann.get("label", template),
                template=template,
                placeholders=ann.get("placeholders", []),
                our_verb=verb,
                video_path=vp,
            ))

    if verbose:
        per_verb = defaultdict(int)
        for c in out:
            per_verb[c.our_verb] += 1
        logger.info("Mapped clips per verb:")
        for v in sorted(per_verb):
            logger.info("  %-10s %d", v, per_verb[v])
        logger.info("Missing video files: %d", n_missing_video)
        logger.info("Distinct unmapped templates: %d (top 10 by count below)",
                    len(seen_unmapped))
        for t, c in sorted(seen_unmapped.items(), key=lambda x: -x[1])[:10]:
            logger.info("  [%5d]  %s", c, t)

    return out
