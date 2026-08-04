#!/usr/bin/env python3
"""Run and report the warped nonmatching contact-patch convergence campaign."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.benchmark_plots import write_line_chart

try:
    from .warped_patch_model import SURFACE_FAMILIES
    from .warped_patch_solver import WarpedPatchCaseRun, solve_warped_patch_case
except ImportError:  # Direct script execution from the repository root.
    from warped_patch_model import SURFACE_FAMILIES
    from warped_patch_solver import WarpedPatchCaseRun, solve_warped_patch_case

SUMMARY_SCHEMA = "contact3d-warped-patch-convergence/v1"
LEVEL_SCHEMA = "contact3d-warped-patch-levels/v1"
RATE_SCHEMA = "contact3d-warped-patch-rates/v1"
BIAS_SCHEMA = "contact3d-warped-patch-bias/v1"
GATE_SCHEMA = "contact3d-warped-patch-gate/v1"
CampaignSolver = Callable[..., WarpedPatchCaseRun]
CaseRequest = tuple[str, str, str]
LEVELS = ("coarse", "medium", "fine")
QUICK_FAMILIES = ("quad-quad",)


class WarpedPatchCampaignError(RuntimeError):
    """Raised after a complete artifact bundle records a failed campaign gate."""


def campaign_families(profile: str) -> tuple[str, ...]:
    """Return the interface matrix exercised by one execution profile."""

    if profile == "quick":
        return QUICK_FAMILIES
    if profile == "full":
        return tuple(SURFACE_FAMILIES)
    raise ValueError("warped patch campaign profile must be quick or full")


def _solve_request(
    solve_case: CampaignSolver,
    request: CaseRequest,
    *,
    publication: bool,
) -> WarpedPatchCaseRun:
    family, bias, level = request
    return solve_case(
        level,
        surface_family=family,
        bias_side=bias,
        publication=publication,
    )


def _solve_campaign_cases(
    profile: str,
    families: Sequence[str],
    *,
    publication: bool,
    solve_case: CampaignSolver,
) -> tuple[WarpedPatchCaseRun, ...]:
    """Solve independent cases in stable order with bounded quick concurrency."""

    requests: tuple[CaseRequest, ...] = tuple(
        (family, bias, level)
        for family in families
        for bias in ("lower", "upper")
        for level in LEVELS
    )
    if profile == "quick" and solve_case is solve_warped_patch_case:
        with ProcessPoolExecutor(max_workers=len(requests)) as executor:
            futures = tuple(
                executor.submit(
                    _solve_request,
                    solve_case,
                    request,
                    publication=publication,
                )
                for request in requests
            )
            return tuple(future.result() for future in futures)
    return tuple(
        _solve_request(solve_case, request, publication=publication)
        for request in requests
    )


def observed_rate(
    coarse_error: float,
    fine_error: float,
    coarse_h: float,
    fine_h: float,
) -> float | None:
    """Return the two-level logarithmic rate, or None for an exact error."""

    values = (coarse_error, fine_error, coarse_h, fine_h)
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("rate inputs must be finite and nonnegative")
    if coarse_h <= fine_h:
        raise ValueError("coarse characteristic size must exceed fine size")
    if coarse_error <= np.finfo(float).tiny or fine_error <= np.finfo(float).tiny:
        return None
    return float(np.log(coarse_error / fine_error) / np.log(coarse_h / fine_h))


def _lookup(
    cases: Sequence[WarpedPatchCaseRun],
) -> dict[tuple[str, str, str], WarpedPatchCaseRun]:
    return {
        (
            str(case.metrics["surface_family"]),
            str(case.metrics["bias_side"]),
            str(case.metrics["profile"]),
        ): case
        for case in cases
    }


def _rate_rows(cases: Sequence[WarpedPatchCaseRun]) -> list[dict[str, object]]:
    lookup = _lookup(cases)
    metrics = (
        "displacement_relative_l2_error",
        "reaction_relative_error",
        "pressure_relative_l2_error",
        "gap_weighted_l2",
        "overlap_area_error",
    )
    rows: list[dict[str, object]] = []
    for family in sorted({key[0] for key in lookup}):
        for bias in ("lower", "upper"):
            selected = [lookup[(family, bias, level)] for level in LEVELS]
            sizes = [float(case.metrics["characteristic_size"]) for case in selected]
            for metric in metrics:
                errors = [float(case.metrics[metric]) for case in selected]
                rows.append(
                    {
                        "surface_family": family,
                        "bias_side": bias,
                        "metric": metric,
                        "coarse_to_medium": observed_rate(
                            errors[0], errors[1], sizes[0], sizes[1]
                        ),
                        "medium_to_fine": observed_rate(
                            errors[1], errors[2], sizes[1], sizes[2]
                        ),
                        "coarse_error": errors[0],
                        "medium_error": errors[1],
                        "fine_error": errors[2],
                    }
                )
    return rows


def _bias_rows(cases: Sequence[WarpedPatchCaseRun]) -> list[dict[str, object]]:
    lookup = _lookup(cases)
    rows: list[dict[str, object]] = []
    for family in sorted({key[0] for key in lookup}):
        for level in LEVELS:
            lower = lookup[(family, "lower", level)].metrics
            upper = lookup[(family, "upper", level)].metrics
            reaction_scale = max(
                float(lower["reference_reaction"]),
                np.finfo(float).tiny,
            )
            pressure_scale = max(
                0.5
                * (
                    float(lower["maximum_pressure"])
                    + float(upper["maximum_pressure"])
                ),
                np.finfo(float).tiny,
            )
            rows.append(
                {
                    "surface_family": family,
                    "profile": level,
                    "characteristic_size": float(lower["characteristic_size"]),
                    "reaction_relative_difference": abs(
                        float(lower["reaction"]) - float(upper["reaction"])
                    )
                    / reaction_scale,
                    "maximum_pressure_relative_difference": abs(
                        float(lower["maximum_pressure"])
                        - float(upper["maximum_pressure"])
                    )
                    / pressure_scale,
                    "overlap_area_difference": abs(
                        float(lower["overlap_area"])
                        - float(upper["overlap_area"])
                    ),
                    "displacement_error_difference": abs(
                        float(lower["displacement_relative_l2_error"])
                        - float(upper["displacement_relative_l2_error"])
                    ),
                }
            )
    return rows


def _gate(
    profile: str,
    levels: Sequence[dict[str, object]],
    bias_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    fine = [row for row in levels if row["profile"] == "fine"]
    criteria = {
        "all_cases_converged": all(bool(row["converged"]) for row in levels),
        "all_cases_reached_final_parameter": all(
            abs(float(row["final_parameter"]) - 1.0) <= 1.0e-12
            for row in levels
        ),
        "normalized_equilibrium_residual": max(
            float(row["normalized_equilibrium_residual"]) for row in levels
        )
        <= 1.1e-8,
        "normalized_penetration": max(
            float(row["normalized_maximum_penetration"]) for row in levels
        )
        <= 2.1e-7,
        "contact_force_balance": max(
            float(row["contact_force_balance_relative"]) for row in levels
        )
        <= 1.0e-10,
        "partition_consistency": max(
            float(row["partition_error"]) for row in levels
        )
        <= 1.0e-10,
        "fine_reaction_accuracy": max(
            float(row["reaction_relative_error"]) for row in fine
        )
        <= 0.15,
        "fine_pressure_accuracy": max(
            float(row["pressure_relative_l2_error"]) for row in fine
        )
        <= 0.35,
        "fine_overlap_accuracy": max(
            float(row["overlap_area_error"]) for row in fine
        )
        <= 0.05,
        "fine_gap_resolution": max(float(row["gap_over_h"]) for row in fine)
        <= 1.0e-4,
        "slave_master_reaction_bias": max(
            float(row["reaction_relative_difference"]) for row in bias_rows
        )
        <= 0.10,
    }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "schema_version": GATE_SCHEMA,
        "profile": profile,
        "passed": not failed,
        "criteria": criteria,
        "failed_criteria": failed,
        "thresholds": {
            "normalized_equilibrium_residual": 1.1e-8,
            "normalized_penetration": 2.1e-7,
            "contact_force_balance": 1.0e-10,
            "partition_consistency": 1.0e-10,
            "fine_reaction_relative_error": 0.15,
            "fine_pressure_relative_l2_error": 0.35,
            "fine_overlap_area_error": 0.05,
            "fine_gap_over_h": 1.0e-4,
            "reaction_bias": 0.10,
        },
    }


def _write_case_fields(
    artifacts: BenchmarkArtifactWriter,
    case: WarpedPatchCaseRun,
) -> tuple[str, ...]:
    root = f"cases/{case.case_id}"
    model = case.model
    step = case.result.accepted_steps[-1]
    contact = step.result.equilibrium.evaluation.contacts[0]
    displacement = step.result.displacement.reshape((-1, 3))[
        model.interface.slave_nodes
    ]
    paths = (
        f"{root}/interface.csv",
        f"{root}/interface.vtp",
        f"{root}/pressure.svg",
        f"{root}/gap.svg",
    )
    artifacts.write_csv(
        paths[0],
        case.interface_rows,
        schema="contact3d-warped-patch-interface/v1",
    )
    artifacts.write_surface_vtp(
        paths[1],
        model.interface.pair.slave.reference_nodes,
        model.interface.pair.slave.facets,
        displacement,
        point_data={
            "normal_gap": contact.normal_gaps,
            "pressure": contact.pressure,
            "multiplier": step.result.states[0].multipliers,
            "active": np.asarray(contact.signature.active_rows, dtype=np.int64),
            "supported": np.asarray(
                contact.signature.supported_rows,
                dtype=np.int64,
            ),
        },
    )
    x_values = np.arange(len(contact.pressure), dtype=float)
    write_line_chart(
        artifacts.output / paths[2],
        title=f"Pressure: {case.case_id}",
        x_label="slave row",
        y_label="pressure",
        x_values=x_values,
        series=((contact.pressure, "mortar pressure"),),
        show_markers=True,
    )
    write_line_chart(
        artifacts.output / paths[3],
        title=f"Gap: {case.case_id}",
        x_label="slave row",
        y_label="normal gap",
        x_values=x_values,
        series=((contact.normal_gaps, "mortar gap"),),
        show_markers=True,
    )
    for path in paths[2:]:
        ElementTree.parse(artifacts.output / path)
        artifacts.register(path, "svg")
    return paths


def _write_convergence_plots(
    artifacts: BenchmarkArtifactWriter,
    cases: Sequence[WarpedPatchCaseRun],
) -> tuple[str, ...]:
    lookup = _lookup(cases)
    (artifacts.output / "convergence").mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    metrics = (
        ("reaction_relative_error", "reaction relative error"),
        ("pressure_relative_l2_error", "pressure relative L2 error"),
        ("gap_weighted_l2", "weighted gap L2"),
    )
    for family in sorted({key[0] for key in lookup}):
        h_values = [
            float(lookup[(family, "upper", level)].metrics["characteristic_size"])
            for level in LEVELS
        ]
        for metric, label in metrics:
            series = tuple(
                (
                    [
                        float(lookup[(family, bias, level)].metrics[metric])
                        for level in LEVELS
                    ],
                    bias,
                )
                for bias in ("lower", "upper")
            )
            path = f"convergence/{family}-{metric}.svg"
            write_line_chart(
                artifacts.output / path,
                title=f"Warped patch {label}: {family}",
                x_label="characteristic size h",
                y_label=label,
                x_values=h_values,
                series=series,
                logarithmic_x=True,
                logarithmic_y=all(min(values) > 0.0 for values, _ in series),
                show_markers=True,
            )
            ElementTree.parse(artifacts.output / path)
            artifacts.register(path, "svg")
            written.append(path)
    return tuple(written)


def run(
    output: Path,
    *,
    profile: str = "quick",
    raise_on_failure: bool = True,
    _solve_case: CampaignSolver = solve_warped_patch_case,
) -> dict[str, object]:
    """Execute one campaign profile and write its complete evidence bundle."""

    families = campaign_families(profile)
    publication = profile == "full"
    artifacts = BenchmarkArtifactWriter(
        output,
        "warped-nonmatching-patch-convergence",
        seed=23017,
        solver_settings={
            "profile": profile,
            "levels": LEVELS,
            "surface_families": families,
            "bias_sides": ("lower", "upper"),
            "publication": publication,
            "parallel_quick_cases": profile == "quick",
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    cases = _solve_campaign_cases(
        profile,
        families,
        publication=publication,
        solve_case=_solve_case,
    )
    levels = [dict(case.metrics) for case in cases]
    rates = _rate_rows(cases)
    bias = _bias_rows(cases)
    gate = _gate(profile, levels, bias)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "profile": profile,
        "case_count": len(cases),
        "surface_families": list(families),
        "levels": list(LEVELS),
        "bias_sides": ["lower", "upper"],
        "passed": bool(gate["passed"]),
        "maximum_reaction_error": max(
            float(row["reaction_relative_error"]) for row in levels
        ),
        "maximum_pressure_error": max(
            float(row["pressure_relative_l2_error"]) for row in levels
        ),
        "maximum_gap_over_h": max(float(row["gap_over_h"]) for row in levels),
        "maximum_reaction_bias": max(
            float(row["reaction_relative_difference"]) for row in bias
        ),
        "total_topology_events": sum(
            int(row["topology_events"]) for row in levels
        ),
        "gate": gate,
    }
    artifacts.write_json("summary.json", summary, schema=SUMMARY_SCHEMA)
    artifacts.write_json("gate.json", gate, schema=GATE_SCHEMA)
    artifacts.write_csv("levels.csv", levels, schema=LEVEL_SCHEMA)
    artifacts.write_csv("rates.csv", rates, schema=RATE_SCHEMA)
    artifacts.write_csv("bias.csv", bias, schema=BIAS_SCHEMA)
    attempts = [
        {"case_id": case.case_id, **row}
        for case in cases
        for row in case.attempt_rows
    ]
    if attempts:
        artifacts.write_csv(
            "attempts.csv",
            attempts,
            schema="contact3d-warped-patch-attempts/v1",
        )
    events = [
        {"case_id": case.case_id, **row}
        for case in cases
        for row in case.event_rows
    ]
    if events:
        artifacts.write_csv(
            "events.csv",
            events,
            schema="contact3d-warped-patch-events/v1",
        )
    required = ["summary.json", "gate.json", "levels.csv", "rates.csv", "bias.csv"]
    for case in cases:
        required.extend(_write_case_fields(artifacts, case))
    required.extend(_write_convergence_plots(artifacts, cases))
    artifacts.finalize(required=required)
    if raise_on_failure and not gate["passed"]:
        raise WarpedPatchCampaignError(
            "warped patch convergence gate failed: "
            + ", ".join(str(value) for value in gate["failed_criteria"])
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/warped-nonmatching-patch-convergence"),
    )
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output, profile=arguments.profile), indent=2))


if __name__ == "__main__":
    main()
