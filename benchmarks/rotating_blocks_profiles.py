"""Deterministic execution profiles for the rotating-blocks benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ProfileName = Literal["quick", "full"]


@dataclass(frozen=True, slots=True)
class RotatingBlocksExecutionProfile:
    """Mesh, path, solver, and evidence settings for one benchmark profile."""

    name: ProfileName
    model_profile: ProfileName
    requested_path_steps: int
    topology_samples: int
    maximum_attempts: int
    initial_step: float
    minimum_step: float
    maximum_step: float
    repetition_runs: int
    refinement_steps: tuple[int, ...]
    export_checkpoints: bool
    run_refinement: bool

    def __post_init__(self) -> None:
        if self.name not in ("quick", "full"):
            raise ValueError("rotating-blocks profile name must be 'quick' or 'full'")
        if self.model_profile != self.name:
            raise ValueError("execution and model profile names must match")
        if self.requested_path_steps < 2:
            raise ValueError("requested path steps must be at least two")
        if self.topology_samples < 9:
            raise ValueError("topology samples must be at least nine")
        if self.maximum_attempts < self.requested_path_steps:
            raise ValueError("maximum attempts must cover requested path steps")
        if not 0.0 < self.minimum_step <= self.initial_step <= self.maximum_step <= 1.0:
            raise ValueError("adaptive step bounds must be positive and ordered")
        if self.repetition_runs < 1:
            raise ValueError("repetition runs must be positive")
        if not self.refinement_steps or any(value < 2 for value in self.refinement_steps):
            raise ValueError("refinement levels must contain path resolutions")
        if tuple(sorted(set(self.refinement_steps))) != self.refinement_steps:
            raise ValueError("refinement levels must be strictly increasing")
        if self.requested_path_steps not in self.refinement_steps:
            raise ValueError("profile path resolution must appear in refinement levels")

    @property
    def path_increment(self) -> float:
        return 1.0 / self.requested_path_steps

    def as_dict(self) -> dict[str, object]:
        """Return a strict machine-readable profile description."""

        return asdict(self)


QUICK_PROFILE = RotatingBlocksExecutionProfile(
    name="quick",
    model_profile="quick",
    requested_path_steps=16,
    topology_samples=65,
    maximum_attempts=128,
    initial_step=1.0 / 16.0,
    minimum_step=1.0 / 1024.0,
    maximum_step=1.0 / 8.0,
    repetition_runs=2,
    refinement_steps=(8, 16, 32),
    export_checkpoints=False,
    run_refinement=False,
)

FULL_PROFILE = RotatingBlocksExecutionProfile(
    name="full",
    model_profile="full",
    requested_path_steps=64,
    topology_samples=129,
    maximum_attempts=1024,
    initial_step=1.0 / 64.0,
    minimum_step=1.0 / 4096.0,
    maximum_step=1.0 / 16.0,
    repetition_runs=2,
    refinement_steps=(32, 64, 128),
    export_checkpoints=True,
    run_refinement=True,
)

PROFILES = {profile.name: profile for profile in (QUICK_PROFILE, FULL_PROFILE)}


def rotating_blocks_execution_profile(
    name: str | RotatingBlocksExecutionProfile,
) -> RotatingBlocksExecutionProfile:
    """Resolve a named profile without changing benchmark semantics."""

    if isinstance(name, RotatingBlocksExecutionProfile):
        return name
    key = str(getattr(name, "name", name))
    try:
        return PROFILES[key]
    except KeyError as error:
        raise ValueError("rotating-blocks profile must be 'quick' or 'full'") from error
