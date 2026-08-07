"""Compatibility façade for event-solver restart diagnostics.

New code should import this API from :mod:`contact3d.solvers.events`.
This module remains only until the compatibility cleanup in #136.
"""

from .solvers.events.restart import (
    RestartAttemptDiagnostic,
    RestartCount,
    RestartDiagnosticOptions,
    RestartDiagnostics,
    RestartEventRecord,
    RestartLoopDiagnostic,
    RestartTerminationReason,
    analyze_restart_diagnostics,
)

__all__ = [
    "RestartAttemptDiagnostic",
    "RestartCount",
    "RestartDiagnosticOptions",
    "RestartDiagnostics",
    "RestartEventRecord",
    "RestartLoopDiagnostic",
    "RestartTerminationReason",
    "analyze_restart_diagnostics",
]
