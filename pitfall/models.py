"""Data models for the pitfall tracking system."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar


class ProblemCategory(Enum):
    """Category of a development problem."""

    ENV_SETUP = "env_setup"
    CUDA_COMPAT = "cuda_compat"
    TRITON_BUG = "triton_bug"
    TEST_FAILURE = "test_failure"
    CODE_GEN = "code_gen"
    PERFORMANCE = "performance"
    OTHER = "other"


class ProblemSeverity(Enum):
    """Severity level of a development problem."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


# Mapping of category tags found in notes/description text to ProblemCategory values.
# Used by both from_markdown_row and the parser.
CATEGORY_TAGS: dict[str, ProblemCategory] = {
    "[ENV]": ProblemCategory.ENV_SETUP,
    "[CUDA]": ProblemCategory.CUDA_COMPAT,
    "[TRITON]": ProblemCategory.TRITON_BUG,
    "[TEST]": ProblemCategory.TEST_FAILURE,
    "[CODEGEN]": ProblemCategory.CODE_GEN,
    "[PERF]": ProblemCategory.PERFORMANCE,
}


def _parse_status_from_text(status_text: str) -> tuple[bool, str]:
    """Parse resolved status from cell text.

    Args:
        status_text: The content of the resolved-status column.

    Returns:
        A tuple of (is_resolved: bool, resolution_note: str).
    """
    status_text = status_text.strip()
    if status_text.startswith("✅"):  # ✅
        return True, status_text[1:].strip()
    elif status_text.startswith("❌"):  # ❌
        return False, status_text[1:].strip()
    else:
        # No emoji found — try to infer from common truthy/falsy patterns.
        lowered = status_text.lower()
        if lowered in ("yes", "true", "1", "y", "done", "fixed", "resolved", "complete"):
            return True, status_text
        return False, status_text


def _infer_category_from_text(text: str) -> ProblemCategory:
    """Infer problem category from text content (description + notes).

    Checks for explicit tag markers (e.g. ``[CUDA]``) first, then falls
    back to keyword matching.

    Args:
        text: Combined text to search for category hints.

    Returns:
        The inferred ProblemCategory, defaulting to OTHER.
    """
    # 1) Explicit tag markers.
    for tag, category in CATEGORY_TAGS.items():
        if tag in text:
            return category

    # 2) Keyword-based heuristics.
    text_lower = text.lower()

    if any(kw in text_lower for kw in ("env", "environment", "pip", "install", "python",
                                        "venv", "virtualenv", "wsl", "nvidia-smi", "driver",
                                        "externally-managed")):
        return ProblemCategory.ENV_SETUP
    if any(kw in text_lower for kw in ("cuda", "nvcc", "sm_", "compute capability",
                                        "nvidia", "gpu driver", "driver update")):
        return ProblemCategory.CUDA_COMPAT
    if any(kw in text_lower for kw in ("triton", "kernel compil", "compil",
                                        "llvm", "cuda_utils", "ptx")):
        return ProblemCategory.TRITON_BUG
    if any(kw in text_lower for kw in ("pytest", "test_", "test failure", "test case",
                                        "assert", "unittest")):
        return ProblemCategory.TEST_FAILURE
    if any(kw in text_lower for kw in ("generation", "codegen", "code_gen", "mask",
                                        "tile", "source code", "kernel gen")):
        return ProblemCategory.CODE_GEN
    if any(kw in text_lower for kw in ("perf", "performance", "slow", "throughput",
                                        "latency", "overhead")):
        return ProblemCategory.PERFORMANCE

    return ProblemCategory.OTHER


@dataclass
class PitfallProblem:
    """A single problem encountered during AI-driven development.

    Attributes:
        number: Sequential problem number (1 = newest).
        timestamp: ISO-8601 date plus hour/minute, e.g. ``2025-06-25 ~14:30``.
        description: Human-readable description of what went wrong.
        repro_steps: Steps to reproduce the problem.
        solution: How the problem was fixed or worked around.
        resolved: Whether the problem has been resolved.
        resolution_note: Short note accompanying the status emoji, e.g. ``WONTFIX``.
        category: Category of the problem (env, triton, test, …).
        severity: How severe the problem is.
        notes: Additional free-form notes.
    """

    number: int
    timestamp: str
    description: str
    repro_steps: str
    solution: str
    resolved: bool
    resolution_note: str = ""
    category: ProblemCategory = ProblemCategory.OTHER
    severity: ProblemSeverity = ProblemSeverity.MINOR
    notes: str = ""

    # ── serialisation ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert this problem to a JSON-serialisable dictionary."""
        return {
            "number": self.number,
            "timestamp": self.timestamp,
            "description": self.description,
            "repro_steps": self.repro_steps,
            "solution": self.solution,
            "resolved": self.resolved,
            "resolution_note": self.resolution_note,
            "category": self.category.value,
            "severity": self.severity.value,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PitfallProblem:
        """Create a problem from a dictionary (inverse of :meth:`to_dict`)."""
        return cls(
            number=data["number"],
            timestamp=data["timestamp"],
            description=data["description"],
            repro_steps=data["repro_steps"],
            solution=data["solution"],
            resolved=data["resolved"],
            resolution_note=data.get("resolution_note", ""),
            category=ProblemCategory(data.get("category", "other")),
            severity=ProblemSeverity(data.get("severity", "minor")),
            notes=data.get("notes", ""),
        )

    # ── markdown table serialisation ───────────────────────────────────

    def to_markdown_row(self) -> str:
        """Format this problem as a single markdown table row.

        The output matches the 7-column schema used in
        ``pitfall-problems-*.log`` files.
        """
        resolved_cell = (
            f"✅ {self.resolution_note}" if self.resolved
            else f"❌ {self.resolution_note}"
        ).strip()
        return (
            f"| {self.number} | {self.timestamp} | {self.description} |"
            f" {self.repro_steps} | {self.solution} | {resolved_cell} |"
            f" {self.notes} |"
        )

    @classmethod
    def from_markdown_row(cls, row: str) -> PitfallProblem:
        """Parse a markdown table row back into a ``PitfallProblem``.

        Expected format (pipe-separated, 7 columns):

            | N | YYYY-MM-DD ~HH:MM | desc | repro | solution | status | notes |
        """
        row = row.strip()
        if not row.startswith("|") or not row.endswith("|"):
            raise ValueError(f"Row must start and end with '|': {row!r}")

        inner = row[1:-1]
        parts = [p.strip() for p in inner.split("|")]

        if len(parts) < 7:
            raise ValueError(f"Expected at least 7 columns, got {len(parts)}: {row!r}")

        number = int(parts[0])
        timestamp = parts[1]
        description = parts[2]
        repro_steps = parts[3]
        solution = parts[4]
        status_text = parts[5]
        notes = "|".join(parts[6:]).strip()  # rejoin extra pipes (edge case)

        resolved, resolution_note = _parse_status_from_text(status_text)
        combined = f"{description} {notes}"
        category = _infer_category_from_text(combined)

        return cls(
            number=number,
            timestamp=timestamp,
            description=description,
            repro_steps=repro_steps,
            solution=solution,
            resolved=resolved,
            resolution_note=resolution_note,
            category=category,
            notes=notes,
        )
