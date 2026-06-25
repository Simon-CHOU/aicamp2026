"""PitfallTracker: main interface for managing pitfall problems."""

from __future__ import annotations

import pathlib
import re
from datetime import datetime
from typing import Any, Optional

from pitfall.models import (
    PitfallProblem,
    ProblemCategory,
    ProblemSeverity,
)
from pitfall.log_parser import parse_log_file

# ---------------------------------------------------------------------------
# Template for the "复用 Prompt" footer appended to every markdown export.
# ---------------------------------------------------------------------------
REUSE_PROMPT_TEMPLATE = """
---
## 复用 Prompt

将此 prompt 给 Claude Code，让它把新问题插入本文件：

```
记录以下问题到 docs/pitfall-problems-{timestamp}.log（若已有今天的文件则追加）：
- 问题描述：[写清楚什么坏了]
- 复现步骤：[做了什么触发]
- 解决方案：[怎么修的，或为什么暂时不修]
- 是否已解决：✅/❌ + 原因

规则：
1. 时间倒序——新问题插到表格第一行（row 1，紧接表头）
2. 已有行序号 # 自动重排
3. 如果问题与已有记录相同，在备注列追加 "→ 复现于 YYYY-MM-DD"
```
"""

# ---------------------------------------------------------------------------
# File header lines (rendered before the table).
# ---------------------------------------------------------------------------
HEADER_LINES = [
    "# Pitfall Problems Log",
    "> 记录 AI agentic coding 过程中遇到的所有问题，倒序排列（最新在前）",
    '> 用途：环境重建参考 + "你遇到过最难的bug是什么"素材',
]

TABLE_COLUMNS = "| # | 时间 | 问题描述 | 复现步骤 | 解决方案 | 已解决 | 备注 |"
TABLE_SEPARATOR = "|---|------|---------|---------|---------|--------|------|"


# ---------------------------------------------------------------------------
# Tracker class
# ---------------------------------------------------------------------------

class PitfallTracker:
    """Tracks development problems in pitfall log files.

    Usage::

        tracker = PitfallTracker()
        tracker.log("description", "repro steps", "solution")
        unresolved = tracker.get_unresolved()
    """

    def __init__(self, log_dir: str | None = None) -> None:
        """Initialise the tracker.

        Args:
            log_dir: Directory containing ``pitfall-problems-*.log`` files.
                     Defaults to ``<pitfall-package-parent>/docs/`` (i.e.
                     ``aicamp2026/docs/`` in the standard layout).
        """
        if log_dir is None:
            self.log_dir = (
                pathlib.Path(__file__).resolve().parent.parent / "docs"
            )
        else:
            self.log_dir = pathlib.Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._problems: list[PitfallProblem] | None = None

    # ── file discovery ──────────────────────────────────────────────────

    def _find_latest_log(self) -> pathlib.Path | None:
        """Return the path of the most recent ``pitfall-problems-*.log``.

        The file with the latest date in its filename wins.  If multiple
        files share the same date, the one with the latest mtime is used.
        """
        files = sorted(
            self.log_dir.glob("pitfall-problems-*.log"),
            key=lambda p: _date_from_filename(p),
            reverse=True,
        )
        return files[0] if files else None

    def latest_log_path(self) -> pathlib.Path | None:
        """Convenience: path to the most recent log file, or ``None``."""
        return self._find_latest_log()

    def today_log_path(self) -> pathlib.Path:
        """Return the path for today's log file.

        The filename follows the convention
        ``pitfall-problems-YYYY-MM-DD.log``.  The file may or may not
        exist yet.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"pitfall-problems-{date_str}.log"

    # ── problem loading ─────────────────────────────────────────────────

    @property
    def problems(self) -> list[PitfallProblem]:
        """Lazily-loaded list of all tracked problems (newest first).

        Reload by calling :meth:`reload` or assigning ``None`` to
        ``_problems``.
        """
        if self._problems is None:
            self._problems = self._load()
        return self._problems

    def _load(self) -> list[PitfallProblem]:
        """Load problems from the latest log file on disk."""
        latest = self._find_latest_log()
        if latest is None:
            return []
        return parse_log_file(str(latest))

    def reload(self) -> None:
        """Force a re-read of the latest log file from disk."""
        self._problems = self._load()

    # ── numbering ───────────────────────────────────────────────────────

    def _next_number(self) -> int:
        """Return the next sequential problem number (1-based)."""
        return len(self.problems) + 1

    def _renumber(self) -> None:
        """Re-number all problems so the newest (index 0) is ``#1``.

        Numbers are assigned consecutively ``1, 2, 3, …`` in list order.
        """
        for idx, problem in enumerate(self.problems):
            problem.number = idx + 1

    # ── file writing ────────────────────────────────────────────────────

    def _write_to_file(self, filepath: pathlib.Path) -> None:
        """Write *all* problems as a complete markdown file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for h in HEADER_LINES:
            lines.append(h)
        lines.append("")
        lines.append(TABLE_COLUMNS)
        lines.append(TABLE_SEPARATOR)
        for p in self.problems:
            lines.append(p.to_markdown_row())
        lines.append("")

        timestamp = datetime.now().strftime("%Y-%m-%d")
        lines.append(REUSE_PROMPT_TEMPLATE.format(timestamp=timestamp).strip())

        content = "\n".join(lines) + "\n"
        filepath.write_text(content, encoding="utf-8")

    # ── core API ────────────────────────────────────────────────────────

    def log(
        self,
        description: str,
        repro_steps: str,
        solution: str,
        resolved: bool = False,
        category: ProblemCategory = ProblemCategory.OTHER,
        severity: ProblemSeverity = ProblemSeverity.MINOR,
        notes: str = "",
    ) -> PitfallProblem:
        """Record a new problem.

        The entry is inserted at the top of the list (newest first),
        renumbered, and persisted to today's log file.

        Args:
            description: What went wrong.
            repro_steps: How to reproduce.
            solution: How it was (or will be) fixed.
            resolved: Whether the issue is already closed.
            category: Problem category.
            severity: Severity level.
            notes: Free-form notes.

        Returns:
            The newly created :class:`PitfallProblem`.
        """
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d ~%H:%M")

        problem = PitfallProblem(
            number=self._next_number(),
            timestamp=timestamp,
            description=description,
            repro_steps=repro_steps,
            solution=solution,
            resolved=resolved,
            category=category,
            severity=severity,
            notes=notes,
        )

        self.problems.insert(0, problem)
        self._renumber()
        self._write_to_file(self.today_log_path())
        return problem

    def query(
        self,
        status: bool | None = None,
        category: ProblemCategory | None = None,
        keyword: str | None = None,
    ) -> list[PitfallProblem]:
        """Query problems by status, category, or keyword.

        Multiple filters are ANDed together.

        Args:
            status: ``True`` = resolved only, ``False`` = unresolved only,
                    ``None`` = both.
            category: Only return problems of this category.
            keyword: Case-insensitive substring match against
                     description, repro_steps, solution, **and** notes.

        Returns:
            A filtered list (newest first).
        """
        results = list(self.problems)

        if status is not None:
            results = [p for p in results if p.resolved == status]

        if category is not None:
            results = [p for p in results if p.category == category]

        if keyword:
            kw = keyword.lower()
            results = [
                p
                for p in results
                if kw in p.description.lower()
                or kw in p.repro_steps.lower()
                or kw in p.solution.lower()
                or kw in p.notes.lower()
            ]

        return results

    def get_unresolved(self) -> list[PitfallProblem]:
        """Shortcut: return every problem that is **not** resolved."""
        return [p for p in self.problems if not p.resolved]

    def resolve(self, number: int, solution: str, note: str = "") -> None:
        """Mark a problem as resolved.

        Args:
            number: Problem number to resolve.
            solution: Updated (or final) solution description.
            note: Short resolution note (e.g. ``"verified"``).

        Raises:
            ValueError: If no problem with the given *number* exists.
        """
        for problem in self.problems:
            if problem.number == number:
                problem.resolved = True
                problem.solution = solution
                problem.resolution_note = note
                self._write_to_file(self.today_log_path())
                return
        raise ValueError(f"Problem #{number} not found")

    # ── statistics ──────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Compute aggregate statistics over all tracked problems.

        Returns a dictionary with keys:

        - ``total`` — total number of problems
        - ``resolved`` — count of resolved problems
        - ``unresolved`` — count of unresolved problems
        - ``by_category`` — mapping ``{category_value: count, …}``
        - ``by_severity`` — mapping ``{severity_value: count, …}``
        """
        total = len(self.problems)
        resolved_count = sum(1 for p in self.problems if p.resolved)
        unresolved_count = total - resolved_count

        by_category: dict[str, int] = {}
        for cat in ProblemCategory:
            by_category[cat.value] = sum(
                1 for p in self.problems if p.category == cat
            )

        by_severity: dict[str, int] = {}
        for sev in ProblemSeverity:
            by_severity[sev.value] = sum(
                1 for p in self.problems if p.severity == sev
            )

        return {
            "total": total,
            "resolved": resolved_count,
            "unresolved": unresolved_count,
            "by_category": by_category,
            "by_severity": by_severity,
        }

    # ── export ──────────────────────────────────────────────────────────

    def export_markdown(self, path: str | None = None) -> str:
        """Export all problems as a complete markdown file.

        Args:
            path: Optional file path.  If given the content is written to
                  disk; otherwise it is returned as a string.

        Returns:
            The markdown string (empty string if *path* was provided).
        """
        self._write_to_file(self.today_log_path())  # ensure consistency

        lines: list[str] = []
        for h in HEADER_LINES:
            lines.append(h)
        lines.append("")
        lines.append(TABLE_COLUMNS)
        lines.append(TABLE_SEPARATOR)
        for p in self.problems:
            lines.append(p.to_markdown_row())
        lines.append("")
        timestamp = datetime.now().strftime("%Y-%m-%d")
        lines.append(REUSE_PROMPT_TEMPLATE.format(timestamp=timestamp).strip())

        result = "\n".join(lines) + "\n"

        if path:
            pathlib.Path(path).write_text(result, encoding="utf-8")
        return result if path is None else ""

    def export_json(self, path: str | None = None) -> str:
        """Export all problems as a JSON array.

        Args:
            path: Optional file path.  If given the content is written to
                  disk; otherwise it is returned as a string.

        Returns:
            The JSON string (empty string if *path* was provided).
        """
        import json

        data = [p.to_dict() for p in self.problems]
        result = json.dumps(data, ensure_ascii=False, indent=2)

        if path:
            pathlib.Path(path).write_text(result, encoding="utf-8")
        return result if path is None else ""


# ── helpers ──────────────────────────────────────────────────────────────

_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _date_from_filename(path: pathlib.Path) -> str:
    """Extract the ISO date string from a ``pitfall-problems-YYYY-MM-DD.log`` filename.

    Falls back to ``"0000-00-00"`` when no date pattern is found.
    """
    match = _DATE_PATTERN.search(path.stem)
    return match.group(1) if match else "0000-00-00"
