"""Shared utilities for eval scripts: model loading, dataset iteration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import yaml

from model import CAPModel


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

    print(f"Loaded ckpt: {ckpt_path.name}  (stage={state.get('stage')}, "
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
