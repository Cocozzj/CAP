"""DEPRECATED — articulation logic moved to eval.baseline.tamp_pddl."""
import warnings
warnings.warn("eval.baseline.tamp_rule.articulation is deprecated; use eval.baseline.tamp_pddl",
                DeprecationWarning, stacklevel=2)

# Re-export for code that still references these names (e.g. eval.baseline.flat_vqvae.infer
# imports apply_pose_trajectory_to_gs and _quat_log_scale_to_full_cov from here).
# These helpers are not TAMP-specific; they live here for legacy reasons.
