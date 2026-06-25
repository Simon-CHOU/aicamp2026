"""Command-line interface for pitfall tracking.

Usage::

    python -m pitfall add "description" "repro" "solution" [--resolved] [--category CODE_GEN]
    python -m pitfall list [--unresolved] [--category CAT] [--keyword text]
    python -m pitfall stats
    python -m pitfall resolve N "solution" [--note "reason"]
    python -m pitfall export [--format json|markdown] [--output path]
"""

from __future__ import annotations

import argparse
import sys

from pitfall import PitfallTracker
from pitfall.models import (
    PitfallProblem,
    ProblemCategory,
    ProblemSeverity,
)


# ── pretty printing ──────────────────────────────────────────────────────

def _category_label(cat: ProblemCategory) -> str:
    """Return a short bracket label for a category, or empty string for OTHER."""
    return f"[{cat.name}]" if cat != ProblemCategory.OTHER else ""


def _severity_label(sev: ProblemSeverity) -> str:
    """Return a severity indicator for non-MINOR severities."""
    return f"({sev.name})" if sev != ProblemSeverity.MINOR else ""


def print_problem(p: PitfallProblem) -> None:
    """Print a single problem in a human-readable format."""
    icon = "✅" if p.resolved else "❌"
    cat = _category_label(p.category)
    sev = _severity_label(p.severity)
    tag = f" {cat} {sev}".strip()

    print(f"{icon}  #{p.number}  {p.timestamp}  {tag}".rstrip())
    print(f"    {p.description}")
    print(f"    Repro:  {p.repro_steps}")
    print(f"    Fix:    {p.solution}")
    if p.resolution_note:
        print(f"    Status: {p.resolution_note}")
    if p.notes:
        print(f"    Notes:  {p.notes}")
    print()


# ── command implementations ──────────────────────────────────────────────

def _cmd_add(args: argparse.Namespace, tracker: PitfallTracker) -> None:
    category = ProblemCategory[args.category]
    severity = ProblemSeverity[args.severity]
    problem = tracker.log(
        description=args.description,
        repro_steps=args.repro,
        solution=args.solution,
        resolved=args.resolved,
        category=category,
        severity=severity,
        notes=args.notes,
    )
    print(f"Added problem #{problem.number}")


def _cmd_list(args: argparse.Namespace, tracker: PitfallTracker) -> None:
    category = ProblemCategory[args.category] if args.category else None

    status_filter: bool | None = None
    if args.unresolved:
        status_filter = False

    results = tracker.query(status=status_filter, category=category, keyword=args.keyword)

    if not results:
        print("No problems found.")
        return

    heading = f"Found {len(results)} problem(s)"
    if args.unresolved:
        heading += " (unresolved only)"
    print(heading)
    print()
    for p in results:
        print_problem(p)


def _cmd_stats(args: argparse.Namespace, tracker: PitfallTracker) -> None:
    s = tracker.stats()
    print(f"Total:        {s['total']}")
    print(f"Resolved:     {s['resolved']}")
    print(f"Unresolved:   {s['unresolved']}")
    print()

    print("By category:")
    for cat, count in sorted(s["by_category"].items()):
        if count > 0:
            print(f"  {cat}: {count}")

    print()
    print("By severity:")
    for sev, count in sorted(s["by_severity"].items()):
        if count > 0:
            print(f"  {sev}: {count}")


def _cmd_resolve(args: argparse.Namespace, tracker: PitfallTracker) -> None:
    try:
        tracker.resolve(args.number, args.solution, note=args.note)
        print(f"Problem #{args.number} resolved.")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_export(args: argparse.Namespace, tracker: PitfallTracker) -> None:
    if args.format == "json":
        output = tracker.export_json(path=args.output)
    else:
        output = tracker.export_markdown(path=args.output)

    if args.output:
        print(f"Exported to {args.output}")
    elif output:
        print(output)


# ── argument parser ──────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pitfall",
        description="Pitfall problem tracker for AI-driven development.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pitfall add "broken tile gen" "size=1000 tile=256" "fix generation.py"
  python -m pitfall list --unresolved
  python -m pitfall list --category CUDA_COMPAT --keyword sm_75
  python -m pitfall stats
  python -m pitfall resolve 5 "Added import sys" --note "verified on ubuntu"
  python -m pitfall export --format json --output /tmp/problems.json
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- add ---
    add = sub.add_parser("add", help="Add a new problem")
    add.add_argument("description", help="What went wrong")
    add.add_argument("repro", help="Steps to reproduce")
    add.add_argument("solution", help="How it was (or will be) fixed")
    add.add_argument("--resolved", action="store_true", help="Mark as resolved")
    add.add_argument(
        "--category",
        default="OTHER",
        choices=[c.name for c in ProblemCategory],
        help="Problem category (default: OTHER)",
    )
    add.add_argument(
        "--severity",
        default="MINOR",
        choices=[s.name for s in ProblemSeverity],
        help="Severity level (default: MINOR)",
    )
    add.add_argument("--notes", default="", help="Additional notes")

    # --- list ---
    lst = sub.add_parser("list", help="List problems")
    lst.add_argument("--unresolved", action="store_true", help="Show only unresolved")
    lst.add_argument(
        "--category",
        default=None,
        choices=[c.name for c in ProblemCategory],
        help="Filter by category",
    )
    lst.add_argument("--keyword", default=None, help="Case-insensitive search")

    # --- stats ---
    sub.add_parser("stats", help="Show aggregate statistics")

    # --- resolve ---
    rslv = sub.add_parser("resolve", help="Mark a problem as resolved")
    rslv.add_argument("number", type=int, help="Problem number (from list)")
    rslv.add_argument("solution", help="Resolution description")
    rslv.add_argument("--note", default="", help="Short resolution note")

    # --- export ---
    export = sub.add_parser("export", help="Export problems")
    export.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        help="Export format (default: markdown)",
    )
    export.add_argument("--output", default=None, help="Output file path")

    return parser


# ── entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point.  Parses ``sys.argv`` and dispatches to sub-commands."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    tracker = PitfallTracker()

    dispatch = {
        "add": _cmd_add,
        "list": _cmd_list,
        "stats": _cmd_stats,
        "resolve": _cmd_resolve,
        "export": _cmd_export,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args, tracker)


if __name__ == "__main__":
    main()
