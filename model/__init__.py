from .model import CAPModel
from .encoder import Encoder
from .planner import Planner
from .executor import Executor
from .utils import (
    CanonicalFrame,
    GSParameter,
    SceneState,
    build_scene_state,
)
from .loss import (
    CAPLoss,
    LossSpec,
    DEFAULT_LOSS_CFG,
    # Individual loss functions (for custom training loops)
    scene_distance,
    closure_loss,
    inverse_loss,
    equivariance_loss,
    equivariance_cross_object_loss,
    commutator_loss,
    reconstruction_loss,
    infonce_loss,
    cvae_loss,
    hierarchical_loss,
    lipschitz_loss,
    entropy_loss,
    physics_loss,
    # Monitoring helpers
    hierarchical_accuracy,
    codebook_utilisation,
)

__all__ = [
    # Core model
    "CAPModel",
    "Encoder",
    "Planner",
    "Executor",
    "SceneState",
    "CanonicalFrame",
    "GSParameter",
    "build_scene_state",
    # Loss
    "CAPLoss",
    "LossSpec",
    "DEFAULT_LOSS_CFG",
    "scene_distance",
    "closure_loss",
    "inverse_loss",
    "equivariance_loss",
    "equivariance_cross_object_loss",
    "commutator_loss",
    "reconstruction_loss",
    "infonce_loss",
    "cvae_loss",
    "hierarchical_loss",
    "lipschitz_loss",
    "entropy_loss",
    "physics_loss",
    "hierarchical_accuracy",
    "codebook_utilisation",
]
