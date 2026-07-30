"""Compressible finite-strain neo-Hookean constitutive model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FloatArray


class BulkGeometryError(ValueError):
    """Raised when a reference or current bulk element is singular or inverted."""


@dataclass(frozen=True, slots=True)
class NeoHookeanMaterial:
    """Compressible neo-Hookean material parameterized by shear and bulk moduli."""

    shear_modulus: float
    bulk_modulus: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.shear_modulus) or self.shear_modulus <= 0.0:
            raise ValueError("shear_modulus must be finite and positive")
        if not np.isfinite(self.bulk_modulus) or self.bulk_modulus <= 0.0:
            raise ValueError("bulk_modulus must be finite and positive")

    @property
    def lame_lambda(self) -> float:
        """Return the first Lamé parameter."""

        return self.bulk_modulus - 2.0 * self.shear_modulus / 3.0

    @classmethod
    def from_young_poisson(
        cls,
        young_modulus: float,
        poisson_ratio: float,
    ) -> NeoHookeanMaterial:
        """Construct a stable isotropic material from Young's modulus and Poisson ratio."""

        if not np.isfinite(young_modulus) or young_modulus <= 0.0:
            raise ValueError("young_modulus must be finite and positive")
        if not np.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie strictly between -1 and 0.5")
        shear = young_modulus / (2.0 * (1.0 + poisson_ratio))
        bulk = young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
        return cls(shear, bulk)


@dataclass(frozen=True, slots=True)
class NeoHookeanResponse:
    """Energy, first Piola stress, and material tangent at one deformation gradient."""

    deformation_gradient: FloatArray
    jacobian: float
    energy_density: float
    first_piola: FloatArray
    tangent: FloatArray


def evaluate_neo_hookean(
    deformation_gradient: FloatArray,
    material: NeoHookeanMaterial,
    *,
    tolerance: float = 1.0e-12,
) -> NeoHookeanResponse:
    """Evaluate the compressible logarithmic neo-Hookean model analytically.

    The strain-energy density is

    ``psi = mu/2 * (F:F - 3) - mu * log(J) + lambda/2 * log(J)^2``.
    """

    deformation = np.asarray(deformation_gradient, dtype=float)
    if deformation.shape != (3, 3):
        raise ValueError("deformation_gradient must have shape (3, 3)")
    if not np.all(np.isfinite(deformation)):
        raise ValueError("deformation_gradient must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    jacobian = float(np.linalg.det(deformation))
    if jacobian <= tolerance:
        raise BulkGeometryError("deformation gradient is singular or inverted")

    mu = material.shear_modulus
    lame = material.lame_lambda
    log_jacobian = float(np.log(jacobian))
    inverse_transpose = np.linalg.inv(deformation).T
    energy_density = (
        0.5 * mu * (float(np.sum(deformation * deformation)) - 3.0)
        - mu * log_jacobian
        + 0.5 * lame * log_jacobian**2
    )
    first_piola = mu * deformation + (lame * log_jacobian - mu) * inverse_transpose

    identity = np.eye(3)
    tangent = (
        mu * np.einsum("ik,jl->ijkl", identity, identity)
        + lame * np.einsum("ij,kl->ijkl", inverse_transpose, inverse_transpose)
        + (mu - lame * log_jacobian)
        * np.einsum("il,kj->ijkl", inverse_transpose, inverse_transpose)
    )
    return NeoHookeanResponse(
        deformation_gradient=deformation.copy(),
        jacobian=jacobian,
        energy_density=float(energy_density),
        first_piola=first_piola,
        tangent=tangent,
    )
