"""Adaptive load continuation, scaling, and penalty-control public API."""

from .adaptive_solver import (
    contact_penalties,
    solve_adaptive_contact_path,
    with_contact_penalties,
)
from .load_path import (
    CoupledLoadPath,
    CoupledPathState,
    LinearBoundaryPath,
    LinearPathValue,
    LoadFactorPath,
    with_coupled_boundary_data,
)
from .rigid_path import RigidBodyBoundaryPath
from .scaling import (
    ContactInterfaceScales,
    ContactScaleIndicators,
    CoupledProblemScales,
    NormalizedKKTDiagnostics,
    PenaltyControlledContactInterface,
    PenaltyUpdateDecision,
    PenaltyUpdatePlan,
    ScaleAwareConvergenceOptions,
    contact_interface_scales,
    coupled_problem_scales,
    propose_interface_penalties,
)
from .solvers import (
    AdaptiveAcceptedStep,
    AdaptiveContactAttempt,
    AdaptiveContactOptions,
    AdaptiveContactResult,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    ScaleAwareAugmentationIteration,
    ScaleAwareAugmentedContactResult,
    ScaleAwareNewtonIteration,
    solve_scale_aware_augmented_contact,
)
from .staged_rigid_path import (
    RigidBodyMotionSegment,
    StagedRigidBodyBoundaryPath,
)

__all__ = [
    "AdaptiveAcceptedStep",
    "AdaptiveContactAttempt",
    "AdaptiveContactOptions",
    "AdaptiveContactResult",
    "AdaptiveLoadOptions",
    "AdaptivePenaltyOptions",
    "ContactInterfaceScales",
    "ContactScaleIndicators",
    "CoupledLoadPath",
    "CoupledPathState",
    "CoupledProblemScales",
    "LinearBoundaryPath",
    "LinearPathValue",
    "LoadFactorPath",
    "NormalizedKKTDiagnostics",
    "PenaltyControlledContactInterface",
    "PenaltyUpdateDecision",
    "PenaltyUpdatePlan",
    "RigidBodyBoundaryPath",
    "RigidBodyMotionSegment",
    "ScaleAwareAugmentationIteration",
    "ScaleAwareAugmentedContactResult",
    "ScaleAwareConvergenceOptions",
    "ScaleAwareNewtonIteration",
    "StagedRigidBodyBoundaryPath",
    "contact_interface_scales",
    "contact_penalties",
    "coupled_problem_scales",
    "propose_interface_penalties",
    "solve_adaptive_contact_path",
    "solve_scale_aware_augmented_contact",
    "with_contact_penalties",
    "with_coupled_boundary_data",
]
