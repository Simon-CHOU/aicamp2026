"""TestEval: comprehensive test evaluation harness for NineToothed T1-2-1.

Runs correctness tests (pytest), checks specialization behaviour, verifies
fallback correctness, inspects generated source structure, and produces a
consolidated report.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any

from benchmark.runner import BenchmarkRunner
from benchmark.schema import TestEvalResult


class TestEval:
    """Evaluates NineToothed changes across all test dimensions.

    Attributes:
        ninetoothed_path: Absolute path to the NineToothed repository.
    """

    def __init__(self, ninetoothed_path: str) -> None:
        """Initialise the evaluation harness.

        Args:
            ninetoothed_path: Path to the root of the NineToothed project.
        """
        self._ninetoothed_path = pathlib.Path(ninetoothed_path)
        self._tests_path = self._ninetoothed_path / "tests"
        self._benchmark_cases_path = (
            pathlib.Path(__file__).resolve().parent / "cases"
        )
        self._last_result: TestEvalResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_pytest(
        self,
        extra_args: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run the NineToothed test suite via pytest.

        Args:
            extra_args: Additional command-line arguments forwarded to
                ``pytest`` (e.g. ``["-x", "-k", "test_add"]``).

        Returns:
            A dict with keys ``total``, ``passed``, ``failed``,
            ``skipped``, and ``output`` (raw stdout/stderr).
        """
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self._tests_path),
            "--tb=short",
            "-q",
            "--no-header",
        ]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute safeguard
        )

        output = result.stdout + result.stderr

        total = 0
        passed = 0
        failed = 0
        skipped = 0

        # Parse summary line(s) of the form:
        #   3 passed, 2 failed, 1 skipped in 1.23s
        #   5 passed in 0.45s
        for line in output.splitlines():
            line = line.strip()
            if "passed" in line and "failed" in line and "in " in line:
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if part.endswith("passed"):
                        passed += _parse_count(part, "passed")
                    elif part.endswith("failed"):
                        failed += _parse_count(part, "failed")
                    elif part.endswith("skipped"):
                        skipped += _parse_count(part, "skipped")
                break
            elif "passed" in line and "in " in line and "failed" not in line:
                passed += _parse_count(line, "passed")
                break

        total = passed + failed + skipped

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "output": output,
        }

    def check_specialization(
        self,
        cases: list | None = None,
    ) -> dict[str, Any]:
        """Verify which cases correctly hit or miss the specialization path.

        Uses ``BenchmarkRunner`` to compile each case and checks whether
        the ``specialization_hit`` flag in the result matches the
        ``specialization_expected`` flag in the case definition.

        Args:
            cases: List of ``BenchmarkCase`` objects.  If ``None``,
                loads the ``SPEC_HIT_CASES`` and ``FALLBACK_CASES`` from
                the registry.

        Returns:
            A dict with keys ``hits``, ``misses``, ``false_hits``, and
            ``details`` (a list of per-case dicts).
        """
        if cases is None:
            from benchmark.cases import ALL_CASES

            cases = ALL_CASES

        runner = BenchmarkRunner(str(self._ninetoothed_path))
        results = runner.run_all(cases)

        hits = 0
        misses = 0
        false_hits = 0
        details: list[dict[str, Any]] = []

        for r in results:
            expected = r.case.specialization_expected
            actual = r.specialization_hit
            entry = {
                "name": r.case.name,
                "expected_specialization": expected,
                "actual_specialization": actual,
                "error": r.error,
            }

            if expected and actual:
                hits += 1
                entry["status"] = "correct_hit"
            elif not expected and not actual:
                hits += 1  # correctly identified as fallback
                entry["status"] = "correct_fallback"
            elif expected and not actual:
                misses += 1
                entry["status"] = "miss"
            else:
                false_hits += 1
                entry["status"] = "false_hit"

            details.append(entry)

        return {
            "hits": hits,
            "misses": misses,
            "false_hits": false_hits,
            "details": details,
        }

    def check_fallback(
        self,
        cases: list | None = None,
    ) -> dict[str, Any]:
        """Verify that fallback cases still produce correct results.

        Compiles each fallback case and validates output against a
        reference implementation (plain PyTorch).

        Args:
            cases: List of ``BenchmarkCase`` objects to test as
                fallbacks.  If ``None``, uses ``FALLBACK_CASES`` from
                the registry.

        Returns:
            A dict with keys ``passed``, ``failed``, and ``details``.
        """
        if cases is None:
            from benchmark.cases import FALLBACK_CASES

            cases = FALLBACK_CASES

        import torch

        runner = BenchmarkRunner(str(self._ninetoothed_path))
        results = runner.run_all(cases)

        passed = 0
        failed = 0
        details: list[dict[str, Any]] = []

        for r in results:
            entry: dict[str, Any] = {
                "name": r.case.name,
                "error": r.error,
            }

            if r.error:
                failed += 1
                entry["passed"] = False
                details.append(entry)
                continue

            # Run correctness check
            dtype = _str_to_torch_dtype(r.case.dtype)
            has_cuda = torch.cuda.is_available()
            device = "cuda" if has_cuda else "cpu"

            try:
                # Re-create inputs and call the compiled kernel to
                # check correctness
                args_list = []
                shape = r.case.input_shape
                args_list.append(torch.rand(*shape, dtype=dtype, device=device))
                args_list.append(torch.rand(*shape, dtype=dtype, device=device))
                output = torch.empty(*shape, dtype=dtype, device=device)
                args_list.append(output)

                # Use jit to compile and get a fresh handle
                _, handle, _ = runner._compile_kernel(r.case)
                handle(*args_list)

                expected = args_list[0] + args_list[1]  # reference
                if torch.allclose(output, expected, atol=1e-4, rtol=1e-3):
                    passed += 1
                    entry["passed"] = True
                else:
                    failed += 1
                    entry["passed"] = False
                    entry["error"] = "Output does not match reference"
            except Exception as exc:
                failed += 1
                entry["passed"] = False
                entry["error"] = str(exc)

            details.append(entry)

        return {
            "passed": passed,
            "failed": failed,
            "details": details,
        }

    def check_generated_source(
        self,
        cases: list | None = None,
    ) -> dict[str, Any]:
        """Check generated Triton source structure against expectations.

        Analyses the generated source for each case and validates that
        it contains expected structural elements (e.g. ``tl.load``,
        ``tl.store``, proper grid definition, no obviously malformed
        AST).

        Args:
            cases: List of ``BenchmarkCase`` objects.  If ``None``,
                uses ``ALL_CASES`` from the registry.

        Returns:
            A dict with keys ``passed``, ``failed``, and ``details``.
        """
        if cases is None:
            from benchmark.cases import ALL_CASES

            cases = ALL_CASES

        runner = BenchmarkRunner(str(self._ninetoothed_path))
        results = runner.run_all(cases)

        passed = 0
        failed = 0
        details: list[dict[str, Any]] = []

        for r in results:
            entry: dict[str, Any] = {
                "name": r.case.name,
                "error": r.error,
            }

            if r.error:
                failed += 1
                entry["passed"] = False
                details.append(entry)
                continue

            cm = r.code_metrics
            if cm is None:
                failed += 1
                entry["passed"] = False
                entry["error"] = "No code metrics available"
                details.append(entry)
                continue

            # Validate structural expectations
            sub = cm.submitted
            issues: list[str] = []

            if sub.source_line_count == 0:
                issues.append("Generated source is empty")
            if sub.mask_expr_count == 0:
                issues.append("No mask expressions found (expected for load/store)")
            if sub.variant_name == "":
                issues.append("No kernel variant name found in source")

            if issues:
                failed += 1
                entry["passed"] = False
                entry["issues"] = issues
            else:
                passed += 1
                entry["passed"] = True
                entry["mask_expr_count"] = sub.mask_expr_count
                entry["stride_expr_count"] = sub.stride_expr_count
                entry["pointer_expr_count"] = sub.pointer_expr_count
                entry["source_line_count"] = sub.source_line_count
                entry["variant_name"] = sub.variant_name

            details.append(entry)

        return {
            "passed": passed,
            "failed": failed,
            "details": details,
        }

    def evaluate(
        self,
        spec_cases: list | None = None,
        fallback_cases: list | None = None,
    ) -> TestEvalResult:
        """Run the full evaluation suite across all dimensions.

        Args:
            spec_cases: Cases for specialisation checking (defaults to
                ``SPEC_HIT_CASES``).
            fallback_cases: Cases for fallback verification (defaults to
                ``FALLBACK_CASES``).

        Returns:
            A ``TestEvalResult`` aggregating all sub-results.
        """
        correctness = self.run_pytest()
        specialization = self.check_specialization(spec_cases)
        fallback = self.check_fallback(fallback_cases)
        source_checks = self.check_generated_source()
        adversarial = {"passed": 0, "failed": 0, "details": []}

        result = TestEvalResult(
            correctness=correctness,
            specialization=specialization,
            fallback=fallback,
            generated_source_checks=source_checks,
            adversarial=adversarial,
        )
        self._last_result = result
        return result

    def report(self) -> str:
        """Generate a human-readable evaluation report.

        Requires that ``evaluate()`` has been called at least once.
        If not, it runs a full evaluation first.

        Returns:
            A multi-line string summarising all evaluation dimensions.
        """
        if self._last_result is None:
            self.evaluate()

        r = self._last_result
        assert r is not None

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  NineToothed T1-2-1  Test Evaluation Report")
        lines.append("=" * 60)

        # Correctness
        corr = r.correctness
        lines.append("")
        lines.append(f"[Correctness]  {corr.get('passed', 0)}/{corr.get('total', 0)}")
        lines.append(f"  Passed:  {corr.get('passed', 0)}")
        lines.append(f"  Failed:  {corr.get('failed', 0)}")
        lines.append(f"  Skipped: {corr.get('skipped', 0)}")

        # Specialization
        spec = r.specialization
        lines.append("")
        lines.append("[Specialization]")
        lines.append(f"  Hits:       {spec.get('hits', 0)}")
        lines.append(f"  Misses:     {spec.get('misses', 0)}")
        lines.append(f"  False hits: {spec.get('false_hits', 0)}")

        if spec.get("details"):
            lines.append("  Details:")
            for d in spec["details"]:
                lines.append(f"    - {d['name']}: {d.get('status', '?')}")

        # Fallback
        fb = r.fallback
        lines.append("")
        lines.append("[Fallback Correctness]")
        lines.append(f"  Passed: {fb.get('passed', 0)}")
        lines.append(f"  Failed: {fb.get('failed', 0)}")
        if fb.get("details"):
            for d in fb["details"]:
                status = "ok" if d.get("passed") else "FAIL"
                err = f" -- {d['error']}" if d.get("error") else ""
                lines.append(f"    - {d['name']}: {status}{err}")

        # Generated-source checks
        gsc = r.generated_source_checks
        lines.append("")
        lines.append("[Generated Source Checks]")
        lines.append(f"  Passed: {gsc.get('passed', 0)}")
        lines.append(f"  Failed: {gsc.get('failed', 0)}")

        # Adversarial
        adv = r.adversarial
        lines.append("")
        lines.append("[Adversarial]")
        lines.append(f"  Passed: {adv.get('passed', 0)}")
        lines.append(f"  Failed: {adv.get('failed', 0)}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


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


def _parse_count(line: str, keyword: str) -> int:
    """Extract a numeric count from a pytest summary fragment.

    Example: ``"3 passed"`` -> ``3``.
    """
    idx = line.find(keyword)
    if idx == -1:
        return 0
    prefix = line[:idx].strip()
    # prefix may contain trailing punctuation etc.
    numbers = [c for c in prefix if c.isdigit()]
    if not numbers:
        return 0
    return int(numbers[0])
