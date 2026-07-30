"""Immutable boundary/load paths for coupled contact continuation."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .coupled import CoupledEquilibriumProblem
from .equilibrium import DeadLoad, DirichletConstraints
from .model import FloatArray, IntArray


def _finite_parameter(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class LinearPathValue:
    """One named scalar interpolated along a linear continuation path."""

    name: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("path-value name must be nonempty")
        if not np.isfinite(self.start) or not np.isfinite(self.end):
            raise ValueError("path-value endpoints must be finite")

    def evaluate(self, parameter: float) -> float:
        return float(self.start + parameter * (self.end - self.start))


@dataclass(frozen=True, slots=True)
class CoupledPathState:
    """Complete immutable boundary snapshot for one continuation parameter."""

    parameter: float
    problem: CoupledEquilibriumProblem
    solver_load_factor: float
    prescribed_dofs: IntArray
    prescribed_values: FloatArray
    effective_force: FloatArray
    values: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        parameter = _finite_parameter(self.parameter, name="path parameter")
        load_factor = _finite_parameter(
            self.solver_load_factor,
            name="solver load factor",
        )
        dofs = np.asarray(self.prescribed_dofs, dtype=np.int64)
        prescribed = np.asarray(self.prescribed_values, dtype=float)
        force = np.asarray(self.effective_force, dtype=float)
        if dofs.ndim != 1 or prescribed.shape != dofs.shape:
            raise ValueError("prescribed path dofs and values must be aligned vectors")
        if np.any(dofs < 0) or len(np.unique(dofs)) != len(dofs):
            raise ValueError("prescribed path dofs must be unique and nonnegative")
        if force.ndim != 1:
            raise ValueError("effective path force must be a flat vector")
        if not np.all(np.isfinite(prescribed)) or not np.all(np.isfinite(force)):
            raise ValueError("path boundary values must be finite")

        normalized_values: list[tuple[str, float]] = []
        names: set[str] = set()
        for name, value in self.values:
            key = str(name).strip()
            number = float(value)
            if not key:
                raise ValueError("path-value name must be nonempty")
            if key in names:
                raise ValueError("path-value names must be unique")
            if not np.isfinite(number):
                raise ValueError("path values must be finite")
            names.add(key)
            normalized_values.append((key, number))

        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "solver_load_factor", load_factor)
        object.__setattr__(self, "prescribed_dofs", dofs.copy())
        constraints = getattr(self.problem, "constraints", None)
        if constraints is not None:
            if not np.array_equal(np.asarray(constraints.dofs, dtype=np.int64), dofs):
                raise ValueError("path snapshot does not match the problem constraint DOFs")
            if not np.array_equal(np.asarray(constraints.values, dtype=float), prescribed):
                raise ValueError("path snapshot does not match the problem prescribed values")
        load = getattr(self.problem, "load", None)
        if load is not None:
            expected_force = load_factor * np.asarray(load.force, dtype=float)
            if not np.array_equal(expected_force, force):
                raise ValueError("path snapshot does not match the problem effective load")

        object.__setattr__(self, "prescribed_values", prescribed.copy())
        object.__setattr__(self, "effective_force", force.copy())
        object.__setattr__(self, "values", tuple(normalized_values))

    @property
    def prescribed_norm(self) -> float:
        return float(np.linalg.norm(self.prescribed_values))

    @property
    def effective_load_norm(self) -> float:
        return float(np.linalg.norm(self.effective_force))

    def value(self, name: str) -> float:
        for key, value in self.values:
            if key == name:
                return value
        raise KeyError(name)

    def with_problem(self, problem: CoupledEquilibriumProblem) -> CoupledPathState:
        """Return the same boundary snapshot attached to an equivalent problem."""

        return CoupledPathState(
            self.parameter,
            problem,
            self.solver_load_factor,
            self.prescribed_dofs,
            self.prescribed_values,
            self.effective_force,
            self.values,
        )


@runtime_checkable
class CoupledLoadPath(Protocol):
    """Evaluate one complete coupled-problem boundary state."""

    def evaluate(
        self,
        problem: CoupledEquilibriumProblem,
        parameter: float,
    ) -> CoupledPathState: ...


def _problem_snapshot(
    problem: CoupledEquilibriumProblem,
    parameter: float,
    solver_load_factor: float,
    values: tuple[tuple[str, float], ...],
) -> CoupledPathState:
    constraints = getattr(problem, "constraints", None)
    load = getattr(problem, "load", None)
    if constraints is None:
        dofs = np.empty(0, dtype=np.int64)
        prescribed = np.empty(0, dtype=float)
    else:
        dofs = np.asarray(constraints.dofs, dtype=np.int64)
        prescribed = np.asarray(constraints.values, dtype=float)
    if load is None:
        force = np.empty(0, dtype=float)
    else:
        force = float(solver_load_factor) * np.asarray(load.force, dtype=float)
    return CoupledPathState(
        parameter,
        problem,
        solver_load_factor,
        dofs,
        prescribed,
        force,
        values,
    )


@dataclass(frozen=True, slots=True)
class LoadFactorPath:
    """Backward-compatible path that scales the problem's existing dead load."""

    value_name: str = "load_factor"

    def __post_init__(self) -> None:
        if not self.value_name or not self.value_name.strip():
            raise ValueError("load-factor path value name must be nonempty")

    def evaluate(
        self,
        problem: CoupledEquilibriumProblem,
        parameter: float,
    ) -> CoupledPathState:
        value = _finite_parameter(parameter, name="path parameter")
        return _problem_snapshot(
            problem,
            value,
            value,
            ((self.value_name, value),),
        )


def with_coupled_boundary_data(
    problem: CoupledEquilibriumProblem,
    constraints: DirichletConstraints,
    load: DeadLoad,
) -> CoupledEquilibriumProblem:
    """Replace boundary data while retaining the exact existing sparsity object.

    The symbolic CSR pattern depends only on the bulk mesh and mapped contact DOFs.
    Boundary values and the dead-load vector may therefore change without rebuilding it.
    """

    total_dofs = 3 * problem.mesh.node_count
    constraints.validate_for(total_dofs)
    if load.force.shape != (total_dofs,):
        raise ValueError("dead-load vector must match the mesh DOF count")
    updated = copy(problem)
    object.__setattr__(updated, "constraints", constraints)
    object.__setattr__(updated, "load", load)
    return updated


@dataclass(frozen=True, slots=True)
class LinearBoundaryPath:
    """Linearly interpolate prescribed displacements, dead loads, and named values."""

    start_constraints: DirichletConstraints
    end_constraints: DirichletConstraints
    start_load: DeadLoad
    end_load: DeadLoad
    values: tuple[LinearPathValue, ...] = ()

    def __post_init__(self) -> None:
        if not np.array_equal(self.start_constraints.dofs, self.end_constraints.dofs):
            raise ValueError("linear path endpoints must constrain the same DOFs")
        if self.start_load.force.shape != self.end_load.force.shape:
            raise ValueError("linear path load endpoints must have equal shapes")
        names = [value.name for value in self.values]
        if len(set(names)) != len(names):
            raise ValueError("linear path-value names must be unique")
        object.__setattr__(self, "values", tuple(self.values))

    @classmethod
    def proportional_prescribed_displacement(
        cls,
        problem: CoupledEquilibriumProblem,
        *,
        values: tuple[LinearPathValue, ...] = (),
    ) -> LinearBoundaryPath:
        start = DirichletConstraints(
            problem.constraints.dofs,
            np.zeros_like(problem.constraints.values),
        )
        return cls(start, problem.constraints, problem.load, problem.load, values)

    @classmethod
    def proportional_dead_load(
        cls,
        problem: CoupledEquilibriumProblem,
        *,
        values: tuple[LinearPathValue, ...] = (),
    ) -> LinearBoundaryPath:
        zero = DeadLoad(np.zeros_like(problem.load.force))
        return cls(problem.constraints, problem.constraints, zero, problem.load, values)

    @classmethod
    def proportional_mixed(
        cls,
        problem: CoupledEquilibriumProblem,
        *,
        values: tuple[LinearPathValue, ...] = (),
    ) -> LinearBoundaryPath:
        start_constraints = DirichletConstraints(
            problem.constraints.dofs,
            np.zeros_like(problem.constraints.values),
        )
        start_load = DeadLoad(np.zeros_like(problem.load.force))
        return cls(
            start_constraints,
            problem.constraints,
            start_load,
            problem.load,
            values,
        )

    def evaluate(
        self,
        problem: CoupledEquilibriumProblem,
        parameter: float,
    ) -> CoupledPathState:
        value = _finite_parameter(parameter, name="path parameter")
        prescribed = self.start_constraints.values + value * (
            self.end_constraints.values - self.start_constraints.values
        )
        force = self.start_load.force + value * (
            self.end_load.force - self.start_load.force
        )
        constraints = DirichletConstraints(self.start_constraints.dofs, prescribed)
        load = DeadLoad(force)
        updated = with_coupled_boundary_data(problem, constraints, load)
        named = tuple((item.name, item.evaluate(value)) for item in self.values)
        return CoupledPathState(
            value,
            updated,
            1.0,
            constraints.dofs,
            constraints.values,
            load.force,
            named,
        )
