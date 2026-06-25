"""Parser for pitfall-problems-*.log markdown files.

The log files follow this structure:

.. code-block:: markdown

    # Pitfall Problems Log
    > 记录 AI agentic coding 过程中遇到的所有问题，倒序排列（最新在前）

    | # | 时间 | 问题描述 | 复现步骤 | 解决方案 | 已解决 | 备注 |
    |---|------|---------|---------|---------|--------|------|
    | 14 | 2025-06-25 ~12:45 | ... | ... | ... | ❌ WONTFIX | ... |
    ...

    ---
    ## 复用 Prompt
    ...
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

from pitfall.models import (
    PitfallProblem,
    ProblemCategory,
    ProblemSeverity,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _is_separator_row(line: str) -> bool:
    """Return True if *line* is a markdown table separator (``|---|---|...``)."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    inner = stripped.replace("|", "").strip()
    return bool(inner) and all(ch in " -" for ch in inner)


def _is_header_row(line: str) -> bool:
    """Return True if *line* looks like the 7-column header row."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    # A header has text content between pipes; a separator has only dashes.
    return not _is_separator_row(stripped) and "|" in stripped


def _parse_status(status_text: str) -> tuple[bool, str]:
    """Parse resolved status from cell text.

    Returns:
        ``(is_resolved: bool, resolution_note: str)``.

    Handles the ``✅`` / ``❌`` emoji markers as well as plain-text English
    status strings.
    """
    status_text = status_text.strip()
    if status_text.startswith("✅"):
        return True, status_text[1:].strip()
    elif status_text.startswith("❌"):
        return False, status_text[1:].strip()
    else:
        lowered = status_text.lower()
        if lowered in ("yes", "true", "1", "y", "done", "fixed", "resolved", "complete"):
            return True, status_text
        return False, status_text


def _infer_category(text: str) -> ProblemCategory:
    """Infer problem category from *text* (description + notes).

    Checks for explicit ``[TAG]`` markers first, then keyword heuristics.

    .. note::

        ``[CODEGEN]`` is intentionally **not** accepted here because the
        existing log uses a different convention.  It can be added back for
        forward compatibility.
    """
    text_lower = text.lower()

    # --- explicit tag markers ---
    tag_map: dict[str, ProblemCategory] = {
        "[ENV]": ProblemCategory.ENV_SETUP,
        "[CUDA]": ProblemCategory.CUDA_COMPAT,
        "[TRITON]": ProblemCategory.TRITON_BUG,
        "[TEST]": ProblemCategory.TEST_FAILURE,
        "[CODEGEN]": ProblemCategory.CODE_GEN,
        "[PERF]": ProblemCategory.PERFORMANCE,
    }
    for tag, cat in tag_map.items():
        if tag in text:
            return cat

    # --- keyword heuristics ---
    if any(kw in text_lower for kw in (
            "env", "environment", "pip", "install", "python",
            "venv", "virtualenv", "wsl", "nvidia-smi", "driver",
            "externally-managed", "pep 668", "segfault")):
        return ProblemCategory.ENV_SETUP

    if any(kw in text_lower for kw in (
            "cuda", "nvcc", "sm_", "compute capability",
            "nvidia", "gpu driver", "driver update", "bf16",
            "sm 7.5", "sm 8.0", "sm 9.0", "sm 12.0")):
        return ProblemCategory.CUDA_COMPAT

    if any(kw in text_lower for kw in (
            "triton", "kernel compil", "compil",
            "llvm", "cuda_utils", "ptx", "map::at")):
        return ProblemCategory.TRITON_BUG

    if any(kw in text_lower for kw in (
            "pytest", "test_", "test failure", "test case",
            "assert", "unittest", "test.py", "test_aot")):
        return ProblemCategory.TEST_FAILURE

    if any(kw in text_lower for kw in (
            "generation", "codegen", "code_gen", "mask",
            "tile", "source code", "kernel gen", "generation.py")):
        return ProblemCategory.CODE_GEN

    if any(kw in text_lower for kw in (
            "perf", "performance", "slow", "throughput",
            "latency", "overhead")):
        return ProblemCategory.PERFORMANCE

    return ProblemCategory.OTHER


# ── public parser ────────────────────────────────────────────────────────

def parse_log_file(filepath: str) -> list[PitfallProblem]:
    """Parse a ``pitfall-problems-*.log`` file and return all problems.

    The parser skips the header lines (``# …``, ``> …``), the column-name
    row, and the separator row.  It stops at the first non-pipe line after
    the table body (typically ``---`` / ``## 复用 Prompt``).

    Args:
        filepath: Absolute or relative path to the log file.

    Returns:
        A list of :class:`PitfallProblem` instances in file order (newest
        first).

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If a data row has fewer than 7 columns or an
            unparseable problem number.
    """
    path = pathlib.Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")

    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")

    problems: list[PitfallProblem] = []
    in_table = False
    continuation: str = ""  # multi-line cell buffer

    for raw_line in lines:
        # --- multi-line continuation ---
        if continuation:
            stripped = raw_line.rstrip()
            if stripped.endswith("\\"):
                continuation += "\n" + stripped[:-1]
                continue
            else:
                continuation += "\n" + stripped
                line = continuation
                continuation = ""
        else:
            line = raw_line

        stripped = line.strip()

        # Skip leading empty lines (before the table).
        if not stripped and not in_table:
            continue

        # --- scan for table header ---
        if not in_table:
            if stripped.startswith("|") and _is_header_row(stripped):
                in_table = True
            continue  # still in preamble

        # --- inside the table ---
        # Skip separator row.
        if _is_separator_row(stripped):
            continue

        # A non-pipe line means we have left the table body.
        if not stripped.startswith("|"):
            break

        # Check for continuation marker (trailing backslash).
        if stripped.endswith("\\"):
            continuation = stripped[:-1]
            continue

        # Parse the data row.
        try:
            problem = PitfallProblem.from_markdown_row(stripped)
        except ValueError as exc:
            raise ValueError(
                f"Failed to parse row in {path.name}: {exc}"
            ) from exc

        problems.append(problem)

    return problems
