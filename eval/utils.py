"""Shared utilities for eval scripts: model loading, dataset iteration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from model import CAPModel
from dataload import DatasetA, DatasetB, collate_batch


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ckpt",       type=str, required=True,
                        help="Path to a stage-end .pt checkpoint")
    parser.add_argument("--config",     type=str, default=None,
                        help="Override config.yaml (default: <ckpt_dir>/../config.yaml)")
    parser.add_argument("--device",     type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write eval outputs (default: <ckpt_dir>/../eval/<mode>)")


def load_model_for_eval(args) -> Tuple[CAPModel, dict, torch.device]:
    """Load CAPModel from checkpoint + matching config; return (model, cfg, device)."""
    ckpt_path = Path(args.ckpt)
    if args.config is not None:
        cfg_path = Path(args.config)
    else:
        cfg_path = ckpt_path.parent.parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found at {cfg_path}.  Pass --config explicitly."
        )

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = CAPModel(cfg).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    print(f"Loaded ckpt: {ckpt_path.name}  (stage={state.get('stage_name')}, "
          f"epoch={state.get('epoch')}, step={state.get('global_step')})")
    return model, cfg, device


def get_output_dir(args, default_subdir: str) -> Path:
    """Figure out where to write eval results."""
    if args.output_dir is not None:
        out = Path(args.output_dir)
    else:
        ckpt_path = Path(args.ckpt)
        out = ckpt_path.parent.parent / "eval" / default_subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


# ══════════════════════════════════════════════════════════════════════
# DatasetA helpers — every eval script uses these to pull a real test split
# ══════════════════════════════════════════════════════════════════════

def add_data_args(parser: argparse.ArgumentParser, *, default_split: str = "test_iid") -> None:
    """Add the standard DatasetA CLI args.  Each eval script picks its own
    ``default_split`` matching the table in the paper:
       - test_iid                 → in-distribution test
       - test_ood_unseen_pair     → new (object, task) combos
       - test_ood_unseen_object   → new objects
       - test_compositional_long  → long-horizon composition
       - val                      → debug / sweep
    """
    parser.add_argument("--dataset",    type=str, default="a", choices=["a", "b"],
                        help="a = DatasetA (3-cam synthetic), b = DatasetB (1-cam real video)")
    parser.add_argument("--manifest",   type=str, required=True,
                        help="data/dataset_<a|b>/manifest.json")
    parser.add_argument("--data-dir",   type=str, required=True,
                        help="data/dataset_<a|b>/data")
    parser.add_argument("--split",      type=str, default=default_split,
                        help=f"dataset split (default: {default_split})")
    parser.add_argument("--T",          type=int, default=30,
                        help="frames per sample (>=10, %%5==0)")
    parser.add_argument("--image-size", type=int, default=256)


def build_eval_loader(
    args,
    sh_dim: int,
    *,
    n_samples:   Optional[int] = None,
    batch_size:  int  = 1,
    num_workers: int  = 0,
    shuffle:     bool = False,
) -> Tuple[object, DataLoader]:
    """Build (truncated) Dataset + matching DataLoader for an eval script.

    Args:
        n_samples:  cap dataset to first N entries (None → use full split)
        batch_size: forwarded to DataLoader
        num_workers: forwarded to DataLoader
        shuffle:    pass True for diversity / sampling-based eval

    Returns:
        (dataset, loader) — some scripts iterate the loader, others poke
        ``dataset[i]`` directly, so we hand back both.
    """
    cls = DatasetA if getattr(args, "dataset", "a") == "a" else DatasetB
    full = cls(
        manifest_path = args.manifest,
        data_dir      = args.data_dir,
        split         = args.split,
        T             = args.T,
        image_size    = args.image_size,
        c_sh          = sh_dim,
    )
    if n_samples is not None and n_samples < len(full):
        ds = Subset(full, range(n_samples))
    else:
        ds = full
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_batch,
        pin_memory=False, drop_last=False,
    )
    return ds, loader
