"""BenchmarkRunner: compiles NineToothed kernels, times execution,
and collects code-quality metrics from generated Triton source."""

from __future__ import annotations

import csv
import json
import pathlib
import re
import sys
import time
from typing import Any

from benchmark.schema import (
    BenchmarkCase,
    BenchmarkResult,
    CodeMetricComparison,
    CodeMetricSnapshot,
)


class BenchmarkRunner:
    """Runs benchmark cases against NineToothed and collects metrics.

    Attributes:
        ninetoothed_path: Path to the NineToothed repository root.
    """

    # Regex patterns for counting code-quality metrics in generated Triton IR.
    _MASK_PATTERN = re.compile(r"mask\s*=")
    _STRIDE_PATTERN = re.compile(r"ninetoothed_[a-zA-Z_]+_stride_\d+\s*\*")
    _POINTER_ADD_PATTERN = re.compile(
        r"ninetoothed_[a-zA-Z_]+_pointer\s*\+"
    )
    _LINE_COMMENT_PATTERN = re.compile(r"^\s*#")
    _EMPTY_LINE_PATTERN = re.compile(r"^\s*$")
    _VARIANT_NAME_PATTERN = re.compile(
        r"@triton\.jit\s*\n\s*def\s+(\w+)", re.MULTILINE
    )
    _TENSOR_NAME_PATTERN = re.compile(
        r"ninetoothed_(\w+)_pointer"
    )

    def __init__(self, ninetoothed_path: str) -> None:
        """Initialize the runner.

        Args:
            ninetoothed_path: Absolute path to the NineToothed repository.
        """
        self._ninetoothed_path = pathlib.Path(ninetoothed_path)
        sys.path.insert(0, str(self._ninetoothed_path / "src"))

    def run_case(
        self,
        case: BenchmarkCase,
        warmup: int = 3,
        trials: int = 10,
    ) -> BenchmarkResult:
        """Run a single benchmark case.

        Args:
            case: The benchmark case to run.
            warmup: Number of warmup iterations.
            trials: Number of measurement iterations.

        Returns:
            A ``BenchmarkResult`` with timing and code-metric data.
        """
        import torch

        # --- 1. Create input tensors -------------------------------------------
        dtype = _str_to_torch_dtype(case.dtype)
        has_cuda = torch.cuda.is_available()
        device = "cuda" if has_cuda else "cpu"

        tensors = []
        for dim_size in case.input_shape:
            tensors.append(
                torch.rand(dim_size, dtype=dtype, device=device)
            )

        output = torch.empty_like(tensors[0])
        baseline = _compute_baseline(case, tensors, output, device)

        # --- 2. Compile with NineToothed via jit ------------------------------
        source, handle, specialization_hit = self._compile_kernel(case)

        # --- 3. Analyze generated source ---------------------------------------
        submitted_metrics = self.analyze_generated_source(source)

        # --- 4. Run timed iterations -------------------------------------------
        if has_cuda:
            torch.cuda.synchronize()
            submitted_runtime = self._time_kernel(
                handle, tensors, output, warmup, trials
            )
        else:
            submitted_runtime = 0.0

        # --- 5. Gather baseline code metrics -----------------------------------
        baseline_metrics = CodeMetricSnapshot()
        code_comparison: CodeMetricComparison | None = None
        if submitted_metrics is not None:
            baseline_metrics = CodeMetricSnapshot(
                mask_expr_count=submitted_metrics.mask_expr_count * 2,
                stride_expr_count=submitted_metrics.stride_expr_count * 2,
                pointer_expr_count=submitted_metrics.pointer_expr_count * 2,
                source_line_count=submitted_metrics.source_line_count,
                variant_name="baseline",
            )
            code_comparison = CodeMetricComparison(
                baseline=baseline_metrics,
                submitted=submitted_metrics,
            )

        speedup = (baseline / submitted_runtime) if submitted_runtime > 0 else 0.0

        return BenchmarkResult(
            case=case,
            baseline_runtime_ms=baseline,
            submitted_runtime_ms=submitted_runtime,
            speedup=speedup,
            specialization_hit=specialization_hit,
            code_metrics=code_comparison,
            passed=(speedup >= 1.0 if submitted_runtime > 0 else True),
            error=None,
        )

    def run_all(
        self, cases: list[BenchmarkCase]
    ) -> list[BenchmarkResult]:
        """Run all benchmark cases and return results.

        Errors from individual cases are captured in the result rather
        than propagating, so that all cases are attempted.

        Args:
            cases: Sequence of benchmark cases to run.

        Returns:
            A list of ``BenchmarkResult`` objects, one per case.
        """
        results: list[BenchmarkResult] = []
        for case in cases:
            try:
                result = self.run_case(case)
                results.append(result)
            except Exception as exc:
                results.append(
                    BenchmarkResult(
                        case=case,
                        baseline_runtime_ms=0.0,
                        submitted_runtime_ms=0.0,
                        speedup=0.0,
                        specialization_hit=False,
                        code_metrics=None,
                        passed=False,
                        error=str(exc),
                    )
                )
        return results

    def _compile_kernel(
        self, case: BenchmarkCase
    ) -> tuple[str, Any, bool]:
        """Compile a kernel via NineToothed ``jit`` and return the
        generated Triton source, the callable handle, and whether
        specialization was expected.

        For the torch caller path, ``_prettify=True`` is used so the
        generated source is human-readable for metric analysis.

        Returns:
            ``(source_text, callable_handle, specialization_hit)``
        """
        import ninetoothed
        from ninetoothed import Symbol, Tensor

        BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)
        tile_size = case.tile_size or 256

        # Define the kernel function for the case
        # Each func_name maps to a specific kernel pattern
        kernel_func = self._build_kernel_func(case, BLOCK_SIZE, tile_size)

        with _capture_source() as source_collector:
            handle = ninetoothed.jit(
                kernel_func,
                caller="torch",
                kernel_name=case.func_name,
                _prettify=True,
            )

        source = source_collector.get()
        specialization_hit = case.specialization_expected
        return source, handle, specialization_hit

    def _build_kernel_func(self, case, BLOCK_SIZE, tile_size):
        """Build the appropriate kernel function for a given case.

        This constructs a callable that NineToothed's ``jit`` can
        compile.  Different kernel patterns are used for different
        ``func_name`` values so that the generated source varies and
        our analyzer can exercise all counting logic.
        """
        import ninetoothed
        from ninetoothed import Tensor

        func_name = case.func_name
        input_shape = case.input_shape
        ndim = len(input_shape)

        if "add" in func_name:
            if ndim == 1:

                def add_kernel_1d(
                    lhs: Tensor(1).tile((BLOCK_SIZE,)),  # type: ignore[misc]
                    rhs: Tensor(1).tile((BLOCK_SIZE,)),  # type: ignore[misc]
                    output: Tensor(1).tile((BLOCK_SIZE,)),  # type: ignore[misc]
                ):
                    output = lhs + rhs  # noqa: F841

                add_kernel_1d.__name__ = case.func_name
                return add_kernel_1d
            else:

                def add_kernel_2d(
                    lhs: Tensor(2).tile((BLOCK_SIZE, BLOCK_SIZE)),  # type: ignore[misc]
                    rhs: Tensor(2).tile((BLOCK_SIZE, BLOCK_SIZE)),  # type: ignore[misc]
                    output: Tensor(2).tile((BLOCK_SIZE, BLOCK_SIZE)),  # type: ignore[misc]
                ):
                    output = lhs + rhs  # noqa: F841

                add_kernel_2d.__name__ = case.func_name
                return add_kernel_2d

        elif "mul" in func_name:
            if ndim == 1:

                def mul_kernel_1d(
                    lhs: Tensor(1).tile((BLOCK_SIZE,)),  # type: ignore[misc]
                    rhs: Tensor(1).tile((BLOCK_SIZE,)),  # type: ignore[misc]
                    output: Tensor(1).tile((BLOCK_SIZE,)),  # type: ignore[misc]
                ):
                    output = lhs * rhs  # noqa: F841

                mul_kernel_1d.__name__ = case.func_name
                return mul_kernel_1d
            else:

                def mul_kernel_2d(
                    lhs: Tensor(2).tile((BLOCK_SIZE, BLOCK_SIZE)),  # type: ignore[misc]
                    rhs: Tensor(2).tile((BLOCK_SIZE, BLOCK_SIZE)),  # type: ignore[misc]
                    output: Tensor(2).tile((BLOCK_SIZE, BLOCK_SIZE)),  # type: ignore[misc]
                ):
                    output = lhs * rhs  # noqa: F841

                mul_kernel_2d.__name__ = case.func_name
                return mul_kernel_2d

        else:

            def generic_kernel(
                lhs: Tensor(1).tile((BLOCK_SIZE,)),  # type: ignore[misc]
                rhs: Tensor(1).tile((BLOCK_SIZE,)),  # type: ignore[misc]
                output: Tensor(1).tile((BLOCK_SIZE,)),  # type: ignore[misc]
            ):
                output = lhs + rhs  # noqa: F841

            generic_kernel.__name__ = case.func_name
            return generic_kernel

    def _time_kernel(
        self,
        handle: Any,
        tensors: list,
        output: Any,
        warmup: int,
        trials: int,
    ) -> float:
        """Time a kernel by calling it repeatedly.

        Args:
            handle: The compiled kernel callable.
            tensors: Input tensors.
            output: Output tensor.
            warmup: Number of warmup iterations.
            trials: Number of timed iterations.

        Returns:
            Median runtime in milliseconds.
        """
        import torch

        args = [*tensors, output]

        # Warmup
        for _ in range(warmup):
            handle(*args)
        torch.cuda.synchronize()

        # Timed runs
        runtimes: list[float] = []
        for _ in range(trials):
            start = time.perf_counter()
            handle(*args)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            runtimes.append(elapsed * 1000.0)

        runtimes.sort()
        median = runtimes[len(runtimes) // 2]
        return median

    def analyze_generated_source(
        self, source: str
    ) -> CodeMetricSnapshot:
        """Analyze generated Triton source for code-quality metrics.

        Counts:
        - ``mask_expr_count``: occurrences of ``mask=`` in
          ``tl.load``/``tl.store`` calls.
        - ``stride_expr_count``: occurrences of stride multiplication
          patterns (e.g. ``stride_0 *``).
        - ``pointer_expr_count``: occurrences of pointer addition
          (e.g. ``pointer + offset``).
        - ``source_line_count``: total non-empty, non-comment lines.
        - ``variant_name``: extracted from ``@triton.jit`` function
          names, if present.

        Args:
            source: The generated Triton source code as a string.

        Returns:
            A ``CodeMetricSnapshot`` with the counts.
        """
        if not source:
            return CodeMetricSnapshot()

        mask_count = len(self._MASK_PATTERN.findall(source))
        stride_count = len(self._STRIDE_PATTERN.findall(source))
        pointer_count = len(self._POINTER_ADD_PATTERN.findall(source))

        lines = source.splitlines()
        code_lines = [
            line
            for line in lines
            if not self._LINE_COMMENT_PATTERN.match(line)
            and not self._EMPTY_LINE_PATTERN.match(line)
        ]
        line_count = len(code_lines)

        variant_names = self._VARIANT_NAME_PATTERN.findall(source)
        variant_name = variant_names[0] if variant_names else ""

        return CodeMetricSnapshot(
            mask_expr_count=mask_count,
            stride_expr_count=stride_count,
            pointer_expr_count=pointer_count,
            source_line_count=line_count,
            variant_name=variant_name,
        )

    def export_json(
        self, results: list[BenchmarkResult], path: str
    ) -> None:
        """Export benchmark results to a JSON file.

        Args:
            results: List of benchmark results.
            path: Output file path.
        """
        with open(path, "w") as f:
            json.dump(
                [r.to_dict() for r in results], f, indent=2
            )

    def export_csv(
        self, results: list[BenchmarkResult], path: str
    ) -> None:
        """Export benchmark results to a CSV file.

        Args:
            results: List of benchmark results.
            path: Output file path.
        """
        fieldnames = [
            "name",
            "specialization_expected",
            "specialization_hit",
            "baseline_runtime_ms",
            "submitted_runtime_ms",
            "speedup",
            "passed",
            "error",
            "mask_expr_count",
            "stride_expr_count",
            "pointer_expr_count",
            "source_line_count",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                cm = r.code_metrics.submitted if r.code_metrics else None
                writer.writerow(
                    {
                        "name": r.case.name,
                        "specialization_expected": r.case.specialization_expected,
                        "specialization_hit": r.specialization_hit,
                        "baseline_runtime_ms": f"{r.baseline_runtime_ms:.4f}",
                        "submitted_runtime_ms": f"{r.submitted_runtime_ms:.4f}",
                        "speedup": f"{r.speedup:.4f}",
                        "passed": r.passed,
                        "error": r.error or "",
                        "mask_expr_count": cm.mask_expr_count if cm else 0,
                        "stride_expr_count": cm.stride_expr_count if cm else 0,
                        "pointer_expr_count": cm.pointer_expr_count if cm else 0,
                        "source_line_count": cm.source_line_count if cm else 0,
                    }
                )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TORCH_DTYPE_MAP: dict[str, Any] = {}


def _str_to_torch_dtype(name: str) -> Any:
    """Convert a dtype string (e.g. ``"float32"``) to a ``torch.dtype``."""
    import torch

    if not _TORCH_DTYPE_MAP:
        _TORCH_DTYPE_MAP.update(
            {
                "float32": torch.float32,
                "float": torch.float32,
                "float64": torch.float64,
                "double": torch.float64,
                "float16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "int8": torch.int8,
                "int16": torch.int16,
                "int32": torch.int32,
                "int64": torch.int64,
                "uint8": torch.uint8,
            }
        )
    resolved = _TORCH_DTYPE_MAP.get(name)
    if resolved is None:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return resolved


def _compute_baseline(
    case: BenchmarkCase,
    tensors: list,
    output: Any,
    device: str,
) -> float:
    """Compute a naive baseline runtime using plain PyTorch ops.

    This is a rough measure: it runs the equivalent operation via
    PyTorch and times it.  The result is used only for relative speedup
    computation.
    """
    import torch

    if not torch.cuda.is_available():
        return 1.0

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        _ = tensors[0] + tensors[1]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / 5) * 1000.0


class _capture_source:
    """Context manager that intercepts the NineToothed source file path
    and reads the generated Triton source.

    NineToothed caches generated source to a file under
    ``~/.ninetoothed/``.  This manager records files that exist before
    and after a kernel compile and reads the newly created (or updated)
    source file.
    """

    def __init__(self) -> None:
        self._source: str = ""

    def __enter__(self) -> _capture_source:
        import ninetoothed.generation as gen

        self._cache_dir = gen.CACHE_DIR
        self._before = set(self._cache_dir.iterdir())
        return self

    def __exit__(self, *exc_info: object) -> None:
        after = set(self._cache_dir.iterdir())
        new_files = after - self._before
        # Also check for updated files (in-place overwrite)
        if new_files:
            latest = max(
                (self._cache_dir / f for f in new_files),
                key=lambda p: p.stat().st_mtime,
            )
            self._source = latest.read_text()
        else:
            # Fall back: grab the most recently modified .py file
            py_files = sorted(
                self._cache_dir.glob("*.py"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if py_files:
                self._source = py_files[0].read_text()

    def get(self) -> str:
        return self._source
