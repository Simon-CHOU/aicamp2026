"""Data models for the NineToothed benchmark and test evaluation infrastructure.

Defines all dataclasses used across the benchmark runner and test evaluation
harness, including serialization to/from JSON.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    """A single benchmark case describing a kernel compilation scenario.

    Attributes:
        name: Unique identifier for the case.
        description: Human-readable description of what the case tests.
        specialization_expected: Whether the AOT compilation is expected to
            select a specialized (divisibility/contiguity-hinted) variant.
        input_shape: Shape of the primary input tensor.
        dtype: Element data type as a string (e.g. ``"float32"``).
        tile_size: The tile size used in the kernel, or ``None`` if not
            applicable.
        func_name: The kernel function name to compile and run.
    """

    name: str
    description: str
    specialization_expected: bool
    input_shape: tuple[int, ...]
    dtype: str
    tile_size: int | None
    func_name: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkCase:
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> BenchmarkCase:
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclasses.dataclass(frozen=True)
class CodeMetricSnapshot:
    """A snapshot of code-quality metrics from generated Triton source.

    Attributes:
        mask_expr_count: Number of ``mask=`` expressions in ``tl.load`` /
            ``tl.store`` calls.
        stride_expr_count: Number of stride-multiplication expressions.
        pointer_expr_count: Number of pointer-arithmetic expressions.
        source_line_count: Total non-empty, non-comment lines.
        variant_name: Name of the kernel variant (if extractable from source).
    """

    mask_expr_count: int = 0
    stride_expr_count: int = 0
    pointer_expr_count: int = 0
    source_line_count: int = 0
    variant_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeMetricSnapshot:
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> CodeMetricSnapshot:
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclasses.dataclass(frozen=True)
class CodeMetricComparison:
    """Comparison between a baseline and a submitted code metric snapshot.

    Attributes:
        baseline: The baseline metric snapshot.
        submitted: The submitted (new) metric snapshot.
        reduction: Fractional reduction for each metric field. A positive
            value means the submitted version improved (fewer
            masks/pointers/stride expressions).
    """

    baseline: CodeMetricSnapshot
    submitted: CodeMetricSnapshot
    reduction: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reduction:
            return
        fields = ("mask_expr_count", "stride_expr_count",
                  "pointer_expr_count", "source_line_count")
        reductions: dict[str, float] = {}
        for field in fields:
            base = getattr(self.baseline, field)
            sub = getattr(self.submitted, field)
            if base > 0:
                reductions[field] = (base - sub) / base
            else:
                reductions[field] = 0.0
        object.__setattr__(self, "reduction", reductions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "submitted": self.submitted.to_dict(),
            "reduction": self.reduction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeMetricComparison:
        return cls(
            baseline=CodeMetricSnapshot.from_dict(data["baseline"]),
            submitted=CodeMetricSnapshot.from_dict(data["submitted"]),
            reduction=data.get("reduction", {}),
        )

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> CodeMetricComparison:
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """Complete result of running a single benchmark case.

    Attributes:
        case: The case that was run.
        baseline_runtime_ms: Baseline runtime in milliseconds.
        submitted_runtime_ms: Submitted (new) runtime in milliseconds.
        speedup: Speedup factor (baseline / submitted). >1 means faster.
        specialization_hit: Whether the AOT specialization path was taken.
        code_metrics: Code metric comparison, or ``None`` if unavailable.
        passed: Whether the case passed all checks.
        error: Error message if something went wrong, or ``None``.
    """

    case: BenchmarkCase
    baseline_runtime_ms: float
    submitted_runtime_ms: float
    speedup: float
    specialization_hit: bool
    code_metrics: CodeMetricComparison | None
    passed: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "case": self.case.to_dict(),
            "baseline_runtime_ms": self.baseline_runtime_ms,
            "submitted_runtime_ms": self.submitted_runtime_ms,
            "speedup": self.speedup,
            "specialization_hit": self.specialization_hit,
            "code_metrics": None,
            "passed": self.passed,
            "error": self.error,
        }
        if self.code_metrics is not None:
            d["code_metrics"] = self.code_metrics.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        cm = data.get("code_metrics")
        return cls(
            case=BenchmarkCase.from_dict(data["case"]),
            baseline_runtime_ms=data["baseline_runtime_ms"],
            submitted_runtime_ms=data["submitted_runtime_ms"],
            speedup=data["speedup"],
            specialization_hit=data["specialization_hit"],
            code_metrics=CodeMetricComparison.from_dict(cm) if cm else None,
            passed=data["passed"],
            error=data.get("error"),
        )

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> BenchmarkResult:
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclasses.dataclass(frozen=True)
class TestEvalResult:
    """Aggregated results from a full test evaluation run.

    Attributes:
        correctness: ``{"total", "passed", "failed", "skipped"}`` counts.
        specialization: ``{"hits", "misses", "false_hits", "details"}``.
        fallback: ``{"passed", "failed", "details"}``.
        generated_source_checks: ``{"passed", "failed", "details"}``.
        adversarial: ``{"passed", "failed", "details"}``.
    """

    correctness: dict[str, Any] = dataclasses.field(default_factory=dict)
    specialization: dict[str, Any] = dataclasses.field(default_factory=dict)
    fallback: dict[str, Any] = dataclasses.field(default_factory=dict)
    generated_source_checks: dict[str, Any] = dataclasses.field(
        default_factory=dict
    )
    adversarial: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestEvalResult:
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> TestEvalResult:
        with open(path) as f:
            return cls.from_dict(json.load(f))
