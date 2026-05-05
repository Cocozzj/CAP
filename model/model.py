"""
CAPModel — top-level orchestrator for CAP-A2GN.

Pipeline
────────
    Encoder  (Stage 1):  frames → slots → motion → VQ tokens + physical_params
    Planner  (Stage 2):  token sequence → task token → CVAE → AR decode
    Executor (Stage 3):  SceneState + physical_params → new SceneState

Training (3-stage curriculum, see PDF f07d2c0a §2.4 / fdfa011c §5.1)
───────────────────────────────────────────────────────────────────
    Stage 0 RIGID    Encoder + Executor.rigid trainable; Planner frozen, physics OFF
    Stage 1 PHYSICS  Executor.deform only; everything else frozen, physics ON
    Stage 2 FULL     all trainable; full loss suite; CVAE β + L_comm annealed

Public API surface
──────────────────
    Construction
        __init__(cfg)
    Training (used by training.py)
        forward(frames, gs_params, ...)         → training_out dict for CAPLoss
        set_stage(stage)                        → flip requires_grad per stage
    Inference modes
        infer_text(texts, scene)                → Mode A
        infer_imitation(demo_frames, gs, scene) → Mode B
        infer_composite(text_list, scene)       → Mode C
        transfer_action(scene, tokens, src, tgt) → cross-object (Prop 3)
    Building blocks (used by eval/)
        encode(frames, gs_params)               → Encoder only
        plan_from_text(texts)                   → Planner only
        plan_composite_from_texts(text_list)    → composite Planner only
        text_to_task(texts)                     → text-only LanguageEncoder lookup
        execute_sequence(scene, physical_params_seq) → Executor only
    Token utilities (used by eval/)
        unflatten_plan(plan, K)
        tokens_to_physical_params(plan_tokens)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import Encoder
from .planner import Planner
from .executor import Executor
from .utils import GSParameter, SceneState, build_scene_state


# ══════════════════════════════════════════════════════════════════════
# Training stage enum
# ══════════════════════════════════════════════════════════════════════

class TrainingStage:
    RIGID   = 0    # Encoder + Executor.rigid (Planner frozen, physics OFF)
    PHYSICS = 1    # Executor.deform only       (rest frozen, physics ON)
    FULL    = 2    # all trainable              (physics ON, all losses on)


# ══════════════════════════════════════════════════════════════════════
# CAPModel
# ══════════════════════════════════════════════════════════════════════

class CAPModel(nn.Module):
    """End-to-end CAP-A2GN model: Encoder + Planner + Executor."""

    # Stage → (encoder trainable, planner trainable, executor trainable, deform-only)
    _STAGE_TABLE: Dict[int, Tuple[bool, bool, bool, bool]] = {
        TrainingStage.RIGID:   (True,  False, True,  False),
        TrainingStage.PHYSICS: (False, False, False, True),
        TrainingStage.FULL:    (True,  True,  True,  False),
    }

    # ────────────────────────────────────────────────────────────────
    # A. Construction
    # ────────────────────────────────────────────────────────────────

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        gs_cfg   = cfg["gs_param"]
        enc_cfg  = cfg["encoder"]
        plan_cfg = cfg["planner"]
        exec_cfg = cfg.get("executor", {})

        # ── Derive K_prim and special token IDs from action codebook ──
        num_action_codes = int(enc_cfg["action_tokenizer"].get("num_action_codebook", 512))
        k_prim = num_action_codes + 1                       # +1 for EOS slot

        # Inject task_dim into all 3 sub-cfgs that need it (single source of truth)
        task_dim = int(plan_cfg["task_dim"])
        plan_cfg["language_encoder_cfg"]["proj_cfg"]["task_dim"] = task_dim
        plan_cfg["task_tokenizer_cfg"]["task_dim"]               = task_dim
        plan_cfg["cvae_cfg"]["task_dim"]                         = task_dim
        plan_cfg["cvae_cfg"]["K_prim"]                           = k_prim

        # Special token IDs (shared)
        specials = dict(plan_cfg["specials"])
        specials["eos_id"] = num_action_codes
        plan_cfg["specials"] = specials
        self._pad_id = int(specials.get("pad_id", -1))
        self._eos_id = int(specials["eos_id"])
        self._num_action_codes = num_action_codes

        # ── Build sub-modules ──
        self.encoder = Encoder(
            gs_dimension=gs_cfg["gs_dimension"],
            obj_cfg=enc_cfg["object_encoder"],
            motion_cfg=enc_cfg["motion_encoder"],
            action_cfg=enc_cfg["action_tokenizer"],
        )
        self.planner = Planner(
            specials=specials,
            sampling_cfg=plan_cfg["sampling_cfg"],
            cvae_cfg=plan_cfg["cvae_cfg"],
            language_encoder_cfg=plan_cfg["language_encoder_cfg"],
            task_tokenizer_cfg=plan_cfg["task_tokenizer_cfg"],
            # Ablation: cfg.planner.use_task_token=False → "w/o hierarchical"
            use_task_token=bool(plan_cfg.get("use_task_token", True)),
        )

        # Executor: physics backend constructor configs from executor.physics
        residual_cfg = exec_cfg.get("residual", {})
        phys_cfg     = exec_cfg.get("physics", {}) or {}
        rc_in        = phys_cfg.get("rigid_contact", {}) or {}
        pbd_in       = phys_cfg.get("pbd", {}) or {}
        rigid_contact_cfg = {}
        if "ground_height" in rc_in: rigid_contact_cfg["ground_z"] = float(rc_in["ground_height"])
        if "gravity"        in rc_in: rigid_contact_cfg["gravity"] = float(rc_in["gravity"])
        rho_parser_cfg = {}
        if "n_iterations" in pbd_in: rho_parser_cfg["n_iters"]    = int(pbd_in["n_iterations"])
        if "n_substeps"   in pbd_in: rho_parser_cfg["n_substeps"] = int(pbd_in["n_substeps"])

        self.executor = Executor(
            rho_dim=int(enc_cfg["action_tokenizer"].get("deformation", {}).get("dim", 16)),
            task_dim=task_dim,
            use_tfn_residual=residual_cfg.get("enabled", True),
            tfn_scalar_dim=int(residual_cfg.get("hidden_scalar", 16)),
            tfn_vector_dim=int(residual_cfg.get("hidden_vector", 4)),
            param_mode=exec_cfg.get("param_mode", "logeuclid"),
            router_temperature=exec_cfg.get("dispatch", {}).get("temperature", 1.0),
            router_hard=exec_cfg.get("dispatch", {}).get("hard", False),
            max_delta_mu=float(exec_cfg.get("max_delta_mu", 1.0)),
            rho_parser_cfg    = rho_parser_cfg    or None,
            rigid_contact_cfg = rigid_contact_cfg or None,
            # Ablation: cfg.executor.physics.enabled_backends → restrict backends
            enabled_backends  = phys_cfg.get("enabled_backends") or None,
        )

        # Materials registry — exposed as model.materials for eval scripts
        self.materials: Dict[str, Dict[str, float]] = exec_cfg.get("materials", {}) or {}

        self._stage = TrainingStage.RIGID

    # ────────────────────────────────────────────────────────────────
    # B. Properties
    # ────────────────────────────────────────────────────────────────

    @property
    def atomic_codebook(self) -> torch.Tensor:
        """Action VQ codebook weight  [K_action, token_dim]."""
        return self.encoder.action_enc.vq.codebook.weight

    @property
    def task_codebook(self) -> torch.Tensor:
        """Task VQ codebook weight  [J, task_dim]."""
        return self.planner.task_codebook_weight()

    @property
    def stage(self) -> int:
        return self._stage

    # ────────────────────────────────────────────────────────────────
    # C. Stage management (data-driven)
    # ────────────────────────────────────────────────────────────────

    def set_stage(self, stage: int) -> None:
        """Flip ``requires_grad`` per the curriculum table.  See ``_STAGE_TABLE``."""
        if stage not in self._STAGE_TABLE:
            raise ValueError(f"Unknown training stage: {stage}")
        enc_train, plan_train, exec_train, deform_only = self._STAGE_TABLE[stage]
        self._stage = stage

        # Encoder
        self._set_requires_grad(self.encoder, enc_train)
        if enc_train and self.encoder.obj_enc.freeze_backbone:
            self._set_requires_grad(self.encoder.obj_enc.backbone, False)

        # Planner (always re-freeze the text encoder backbone if planner is on)
        self._set_requires_grad(self.planner, plan_train)
        if plan_train:
            self._set_requires_grad(self.planner.lang.text_enc, False)

        # Executor: special PHYSICS-stage handling ─ only deform branch trainable
        if deform_only:
            self._set_requires_grad(self.executor, False)
            if hasattr(self.executor, "deform"):
                self._set_requires_grad(self.executor.deform, True)
        else:
            self._set_requires_grad(self.executor, exec_train)

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for p in module.parameters():
            p.requires_grad = requires_grad

    # ────────────────────────────────────────────────────────────────
    # D. Training forward (one method)
    # ────────────────────────────────────────────────────────────────

    def forward(
        self,
        frames:    torch.Tensor,                   # [B, V, T, C, H, W]
        gs_params: List[GSParameter],              # len B
        tau: float = 1.0,
        condition: Optional[Dict[str, Any]] = None,
        enable_physics: Optional[bool] = None,     # None → derive from stage
    ) -> Dict[str, Any]:
        """Unified training forward — runs Encoder → Planner → Executor.

        Returns the ``training_out`` dict consumed by ``CAPLoss``::

            {
              "encoder":       full Encoder output (incl. physical_params)
              "planner":       full Planner training output
              "executor":      {final_state, trajectory, aux_list}
              "scene_state":   initial SceneState
              "token_indices": [B, L] flat AR target tokens
            }
        """
        if enable_physics is None:
            enable_physics = (self._stage >= TrainingStage.PHYSICS)
        cond = condition or {}

        # ── Encoder (Stage 1) ─────────────────────────────────────
        enc_out = self.encoder(frames, tau=tau, gs_params=gs_params)

        # ── Flatten token grid → 1D sequence for Planner ─────────
        B = enc_out["seq_tokens"].size(0)
        token_indices = enc_out["seq_tokens"].reshape(B, -1).clone()
        if enc_out["seq_mask"] is not None:
            flat_mask = enc_out["seq_mask"].reshape(B, -1)
            token_indices[~flat_mask] = self._pad_id

        # ── Planner (Stage 2) ─────────────────────────────────────
        plan_out = self.planner.training_forward(
            token_indices=token_indices,
            atomic_codebook=self.atomic_codebook,
            text_labels=cond.get("texts"),
            sample_prob=float(cond.get("sample_prob", 0.0)),
            deterministic=bool(cond.get("deterministic", False)),
        )

        # ── SceneState from gs_params + Encoder's phi ─────────────
        scene_state = build_scene_state(
            gs_params=gs_params,
            phi=enc_out["phi"],
            assignment=enc_out["assignment"],
        )

        # ── Executor (Stage 3) — direct physical_params from Encoder
        task_context = self._expand_task_context(plan_out["task_emb"], scene_state.K)
        exec_out = self.execute_sequence(
            scene=scene_state,
            physical_params_seq=enc_out["physical_params"],
            enable_physics=enable_physics,
            task_context=task_context,
        )

        return {
            "encoder":       enc_out,
            "planner":       plan_out,
            "executor":      exec_out,
            "scene_state":   scene_state,
            "token_indices": token_indices,
        }

    # ────────────────────────────────────────────────────────────────
    # E. Inference modes — Mode A / B / C + cross-object
    # ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def infer_text(
        self,
        texts: List[str],
        scene: SceneState,
        sampling_info: Optional[Dict[str, Any]] = None,
        enable_physics: bool = True,
        task_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Mode A — text → action plan → execute on scene."""
        plan_out = self.plan_from_text(texts, sampling_info=sampling_info, num_samples=1)
        K = scene.K
        plan_tokens = self.unflatten_plan(plan_out["sequences"], K=K)
        physical_params_seq = self.tokens_to_physical_params(plan_tokens)

        if task_context is None:
            task_context = self._expand_task_context(plan_out["task_emb"], K)

        exec_out = self.execute_sequence(
            scene=scene,
            physical_params_seq=physical_params_seq,
            enable_physics=enable_physics,
            task_context=task_context,
        )
        return {
            "plan":        plan_out,
            "plan_tokens": plan_tokens,
            **exec_out,                            # final_state, trajectory, aux_list
        }

    @torch.no_grad()
    def infer_imitation(
        self,
        demo_frames: torch.Tensor,                 # [B, V, T, C, H, W]
        gs_params: List[GSParameter],              # len B
        scene: SceneState,
        tau: float = 1.0,
        enable_physics: bool = True,
        task_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Mode B — encode demonstration, replay physical_params on a new scene."""
        enc_out = self.encoder(demo_frames, tau=tau, gs_params=gs_params)
        exec_out = self.execute_sequence(
            scene=scene,
            physical_params_seq=enc_out["physical_params"],
            enable_physics=enable_physics,
            task_context=task_context,
        )
        return {
            "encoder":    enc_out,
            "seq_tokens": enc_out["seq_tokens"],
            **exec_out,
        }

    @torch.no_grad()
    def infer_composite(
        self,
        text_list: List[List[str]],
        scene: SceneState,
        sampling_info: Optional[Dict[str, Any]] = None,
        enable_physics: bool = True,
        task_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Mode C — composite task: each sub-task gets its own task context."""
        plan_out = self.plan_composite_from_texts(text_list, sampling_info=sampling_info)
        K = scene.K
        all_traj: List[SceneState] = []
        all_aux:  List[dict]       = []

        for sub_seq, t_emb in zip(plan_out["sub_seqs"], plan_out["task_embs"]):
            sub_tokens = self.unflatten_plan(sub_seq, K=K)
            if sub_tokens.shape[1] == 0:
                continue
            tc = task_context if task_context is not None \
                 else self._expand_task_context(t_emb, K)
            exec_out = self.execute_sequence(
                scene=scene,
                physical_params_seq=self.tokens_to_physical_params(sub_tokens),
                enable_physics=enable_physics,
                task_context=tc,
            )
            scene = exec_out["final_state"]
            all_traj.extend(exec_out["trajectory"])
            all_aux.extend(exec_out["aux_list"])

        return {
            "plan":        plan_out,
            "plan_tokens": self.unflatten_plan(plan_out["full_seq"], K=K),
            "final_state": scene,
            "trajectory":  all_traj,
            "aux_list":    all_aux,
        }

    @torch.no_grad()
    def transfer_action(
        self,
        scene: SceneState,
        token_indices: torch.Tensor,                # [B, K] single-step
        src_k: int,
        tgt_k: int,
        enable_physics: bool = True,
        task_context: Optional[torch.Tensor] = None,
    ) -> Tuple[SceneState, dict]:
        """Cross-object transfer (Proposition 3): apply src's action on tgt."""
        # [B, K] → [B, 1, K] → physical_params dict → strip T
        params_3d = self.tokens_to_physical_params(token_indices.unsqueeze(1))
        params_2d = {k: (v.squeeze(1) if v is not None else None) for k, v in params_3d.items()}
        return self.executor.transfer_object(
            scene=scene, physical_params=params_2d,
            src_k=src_k, tgt_k=tgt_k,
            enable_physics=enable_physics, task_context=task_context,
        )

    # ────────────────────────────────────────────────────────────────
    # F. Public building blocks — used by eval scripts
    # ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        frames: torch.Tensor,
        gs_params: List[GSParameter],
        tau: float = 1.0,
    ) -> Dict[str, Any]:
        """Run only the Encoder (data prep / eval)."""
        return self.encoder(frames, tau=tau, gs_params=gs_params)

    @torch.no_grad()
    def plan_from_text(
        self,
        texts: List[str],
        sampling_info: Optional[Dict[str, Any]] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run only Planner.sample_actions on text prompts."""
        return self.planner.sample_actions(
            texts=texts, sampling_info=sampling_info, num_samples=num_samples,
        )

    @torch.no_grad()
    def plan_composite_from_texts(
        self,
        text_list: List[List[str]],
        sampling_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Plan a composite (multi-step) task from sub-instructions."""
        task_embs = [self.planner.infer_task_from_text(t)["task_emb"] for t in text_list]
        plan_out = self.planner.plan_composite(
            task_embs=task_embs, sampling_info=sampling_info,
        )
        plan_out["task_embs"] = task_embs
        return plan_out

    @torch.no_grad()
    def text_to_task(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Map natural language → nearest task codebook entry."""
        return self.planner.infer_task_from_text(texts)

    def execute_sequence(
        self,
        scene: SceneState,
        physical_params_seq: Dict[str, torch.Tensor],   # [B, T, K, ...]
        enable_physics: bool = False,
        task_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run only the Executor over a sequence of physical_params.

        Returns: ``{final_state, trajectory, aux_list}``.

        NOT decorated with ``@torch.no_grad`` because ``forward`` needs to
        backprop through it; eval scripts can wrap their own no_grad context.
        """
        final_state, trajectory, aux_list = self.executor.apply_sequence(
            scene=scene,
            physical_params_seq=physical_params_seq,
            enable_physics=enable_physics,
            task_context=task_context,
        )
        return {
            "final_state": final_state,
            "trajectory":  trajectory,
            "aux_list":    aux_list,
        }

    # ────────────────────────────────────────────────────────────────
    # G. Public token utilities — used by eval scripts
    # ────────────────────────────────────────────────────────────────

    def unflatten_plan(
        self,
        plan: torch.Tensor,                             # [B, L_out]
        K: int,
    ) -> torch.Tensor:
        """[B, L] flat plan → [B, T, K] structured tokens.

        Trims trailing tokens that don't fill a complete (T, K) frame; replaces
        EOS and post-EOS positions with the last valid action token; clamps
        out-of-range to a valid codebook index.
        """
        B, L = plan.shape
        T = L // K

        if T == 0:
            # Pad to at least one (T=1) timestep
            pad_len = K - L
            tokens  = torch.cat([plan, plan[:, -1:].expand(B, pad_len)], dim=1).clone()
            T, usable = 1, K
        else:
            usable = T * K
            tokens = plan[:, :usable].clone()

        # Replace EOS and post-EOS with last-valid action token (per row)
        for b in range(B):
            eos_mask = (tokens[b] == self._eos_id)
            if eos_mask.any():
                first_eos  = eos_mask.nonzero(as_tuple=True)[0][0].item()
                last_valid = tokens[b, first_eos - 1].item() if first_eos > 0 else 0
                tokens[b, first_eos:] = last_valid

        return tokens.reshape(B, T, K).clamp(0, self._num_action_codes - 1)

    def tokens_to_physical_params(
        self,
        plan_tokens: torch.Tensor,                      # [B, T, K] long
    ) -> Dict[str, torch.Tensor]:
        """Decode token indices → structured physical_params for the Executor."""
        return self.encoder.action_enc.tokens_to_physical_params(plan_tokens)

    # ────────────────────────────────────────────────────────────────
    # H. Private helpers
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _expand_task_context(
        task_emb: torch.Tensor,                         # [B, task_dim]
        K: int,
    ) -> torch.Tensor:
        """[B, task_dim] → [B, K, task_dim] (broadcast per-object)."""
        return task_emb.unsqueeze(1).expand(-1, K, -1).contiguous()

    # ────────────────────────────────────────────────────────────────
    # I. PyTorch override
    # ────────────────────────────────────────────────────────────────

    def train(self, mode: bool = True):
        """Propagate train/eval mode while respecting frozen sub-modules.

        ObjectDecomposer / LanguageEncoder handle their own backbone freezing
        in their .train() overrides; stage-based requires_grad is unaffected.
        """
        super().train(mode)
        return self
