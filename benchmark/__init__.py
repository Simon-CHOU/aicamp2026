"""Benchmark and test evaluation infrastructure for NineToothed T1-2-1.

Provides:
- BenchmarkRunner: compile, time, and collect metrics for benchmark cases.
- TestEval: comprehensive test evaluation harness.
- Data models: BenchmarkCase, BenchmarkResult, CodeMetricSnapshot,
  CodeMetricComparison, TestEvalResult.
"""

from benchmark.schema import (
    BenchmarkCase,
    BenchmarkResult,
    CodeMetricComparison,
    CodeMetricSnapshot,
    TestEvalResult,
)
from benchmark.runner import BenchmarkRunner
from benchmark.testeval import TestEval

__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "CodeMetricComparison",
    "CodeMetricSnapshot",
    "TestEvalResult",
    "BenchmarkRunner",
    "TestEval",
]
