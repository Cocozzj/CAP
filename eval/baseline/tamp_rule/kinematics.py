"""DEPRECATED — kinematic helpers moved to eval.baseline.kinematics."""
import warnings
warnings.warn("eval.baseline.tamp_rule.kinematics moved to eval.baseline.kinematics",
                DeprecationWarning, stacklevel=2)
from ..kinematics import (   # noqa: F401, E402
    apply_pose_trajectory_to_gs,
    quat_xyzw_to_R,
    quat_log_scale_to_full_cov,
)
