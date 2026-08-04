"""User-facing configuration for the sandwiched-beam example."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from benchmarks.sandwiched_beam_model import (
    SandwichedBeamGeometry,
    SandwichedBeamModel,
    build_sandwiched_beam_model,
)
from contact3d import (
    ContactPair,
    ContactSurface,
    MortarContactInterface,
    Tet4Mesh,
)

LEVEL = "coarse"
SLAVE_SIDE = "upper"
TOPOLOGY_OFFSET = 1.0e-4
NORMAL_PENALTY = 200.0
END_MOMENT = 0.02


def _regularize_initial_topology(model: SandwichedBeamModel) -> SandwichedBeamModel:
    """Move one body in-plane so the initial clipping branch is unambiguous."""

    nodes = model.problem.mesh.reference_nodes.copy()
    upper_start = len(model.lower_nodes)
    shift = np.asarray((TOPOLOGY_OFFSET, TOPOLOGY_OFFSET, 0.0))
    nodes[upper_start:] += shift

    interface = model.problem.interfaces[0]
    pair = interface.pair
    slave = ContactSurface(
        nodes[interface.slave_nodes],
        pair.slave.facets,
        normal_sign=pair.slave.normal_sign,
    )
    master = ContactSurface(
        nodes[interface.master_nodes],
        pair.master.facets,
        normal_sign=pair.master.normal_sign,
    )
    shifted_pair = ContactPair(
        slave,
        master,
        normal_penalty=NORMAL_PENALTY,
        search_distance=pair.search_distance,
        quadrature_points=pair.quadrature_points,
    )
    shifted_interface = MortarContactInterface(
        shifted_pair,
        interface.slave_nodes,
        interface.master_nodes,
    )
    mesh = Tet4Mesh(nodes, model.problem.mesh.elements)
    problem = replace(
        model.problem,
        mesh=mesh,
        interfaces=(shifted_interface,),
    )
    return replace(
        model,
        problem=problem,
        upper_nodes=model.upper_nodes + shift,
    )


def build_model() -> SandwichedBeamModel:
    """Build the one modest nonmatching model used by the v0.1 example."""

    model = build_sandwiched_beam_model(
        LEVEL,
        slave_side=SLAVE_SIDE,
        geometry=SandwichedBeamGeometry(end_moment=END_MOMENT),
    )
    return _regularize_initial_topology(model)


__all__ = [
    "END_MOMENT",
    "LEVEL",
    "NORMAL_PENALTY",
    "SLAVE_SIDE",
    "TOPOLOGY_OFFSET",
    "SandwichedBeamGeometry",
    "SandwichedBeamModel",
    "build_model",
]
