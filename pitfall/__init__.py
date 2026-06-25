"""Pitfall problem tracking system for AI-driven development.

Provides:
- PitfallTracker: add, query, and manage development problems.
- PitfallProblem: data model for a single problem entry.
- parse_log_file: parse existing pitfall-problems-*.log markdown files.
"""

from pitfall.models import PitfallProblem, ProblemCategory, ProblemSeverity
from pitfall.tracker import PitfallTracker
from pitfall.log_parser import parse_log_file

__all__ = [
    "PitfallProblem",
    "ProblemCategory",
    "ProblemSeverity",
    "PitfallTracker",
    "parse_log_file",
]
