"""Phase checkpoint definitions for the T1-2-1 development loop.

Each checkpoint is a function that takes the current state context
(test results, benchmark data, pitfall stats) and returns whether
the phase exit criteria are met.

Checkpoints are the GATE between phases — the loop engine will not
advance to the next phase until the current phase's checkpoint passes.
"""

from typing import Callable, Any

# Type alias for checkpoint functions
# Takes (testeval_result, benchmark_result, pitfall_stats, context) -> (passed, blockers, suggestions)
CheckpointFn = Callable[
    [dict | None, dict | None, dict | None, dict | None],
    tuple[bool, list[str], list[str]],
]


def checkpoint_phase_0(
    testeval: dict | None = None,
    benchmark: dict | None = None,
    pitfall_stats: dict | None = None,
    context: dict | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Phase 0: Loop engineering infrastructure setup.

    Checks: benchmark/, pitfall/, loop/ packages importable;
    pitfall can parse existing .log; loop engine initializes.
    """
    blockers = []
    suggestions = []

    # Check that packages are importable
    try:
        import benchmark  # noqa: F401
    except ImportError:
        blockers.append("benchmark/ package not importable")

    try:
        import pitfall  # noqa: F401
    except ImportError:
        blockers.append("pitfall/ package not importable")

    try:
        import loop  # noqa: F401
    except ImportError:
        blockers.append("loop/ package not importable")

    # Check pitfall can parse logs
    if pitfall_stats is None or pitfall_stats.get("total", 0) == 0:
        suggestions.append("No pitfall entries parsed — verify log parser works")

    return (len(blockers) == 0, blockers, suggestions)


def checkpoint_phase_1(
    testeval: dict | None = None,
    benchmark: dict | None = None,
    pitfall_stats: dict | None = None,
    context: dict | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Phase 1: Environment setup and baseline confirmation.

    Checks: pytest all pass (except known WONTFIX), CUDA available,
    Triton importable, baseline metrics recorded.
    """
    blockers = []
    suggestions = []

    if testeval is None:
        blockers.append("No testeval results — run pytest first")
    else:
        if testeval.get("failed", 0) > 0:
            # Allow known WONTFIX failures
            known_wontfix = testeval.get("known_wontfix", [])
            actual_failures = testeval.get("failed", 0) - len(known_wontfix)
            if actual_failures > 0:
                blockers.append(
                    f"pytest has {actual_failures} unexpected failures "
                    f"(total {testeval['failed']} failed, "
                    f"{len(known_wontfix)} known WONTFIX)"
                )

        if testeval.get("passed", 0) == 0:
            blockers.append("No tests passed — environment likely broken")

    # Check CUDA
    cuda_available = context.get("cuda_available", False) if context else False
    if not cuda_available:
        suggestions.append(
            "CUDA not confirmed — verify torch.cuda.is_available()"
        )

    # Check Triton
    triton_available = context.get("triton_available", False) if context else False
    if not triton_available:
        blockers.append("Triton not importable — cannot compile kernels")

    # Check baselines
    if benchmark is None or not benchmark.get("baselines_recorded", False):
        suggestions.append("Baseline metrics not yet recorded")

    return (len(blockers) == 0, blockers, suggestions)


def checkpoint_phase_2(
    testeval: dict | None = None,
    benchmark: dict | None = None,
    pitfall_stats: dict | None = None,
    context: dict | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Phase 2: Weakness analysis.

    Checks: >=2 weakness cases analyzed with quantified metrics,
    weakness_analysis.md written, is_contiguous prototype verified.
    """
    blockers = []
    suggestions = []

    weakness_cases = context.get("weakness_cases", 0) if context else 0
    if weakness_cases < 2:
        blockers.append(
            f"Only {weakness_cases}/2 weakness cases analyzed"
        )

    analysis_doc_exists = context.get("weakness_analysis_doc", False) if context else False
    if not analysis_doc_exists:
        blockers.append("docs/weakness_analysis.md not found")

    prototype_verified = context.get("prototype_verified", False) if context else False
    if not prototype_verified:
        suggestions.append(
            "is_contiguous prototype not verified — "
            "may need to do this before Phase 3 design"
        )

    # Check quantified metrics
    has_metrics = context.get("baseline_metrics", False) if context else False
    if not has_metrics:
        suggestions.append("Quantified baseline metrics not recorded for all cases")

    return (len(blockers) == 0, blockers, suggestions)


def checkpoint_phase_3(
    testeval: dict | None = None,
    benchmark: dict | None = None,
    pitfall_stats: dict | None = None,
    context: dict | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Phase 3: Specialization category selection and design.

    Checks: 1-2 categories selected, boolean predicates written,
    fallback path documented, no hardcoding.
    """
    blockers = []
    suggestions = []

    categories_selected = context.get("categories_selected", 0) if context else 0
    if categories_selected < 1:
        blockers.append("No specialization categories selected")

    predicates_written = context.get("predicates_written", False) if context else False
    if not predicates_written:
        blockers.append("Boolean predicates not written for specialization conditions")

    fallback_documented = context.get("fallback_documented", False) if context else False
    if not fallback_documented:
        blockers.append("Fallback path not documented")

    has_hardcoding = context.get("has_hardcoding", True) if context else True
    if has_hardcoding:
        blockers.append(
            "Design contains hardcoded dimensions, filenames, or benchmark names"
        )

    return (len(blockers) == 0, blockers, suggestions)


def checkpoint_phase_4(
    testeval: dict | None = None,
    benchmark: dict | None = None,
    pitfall_stats: dict | None = None,
    context: dict | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Phase 4: Implementation.

    Checks: pytest all pass, fallback path verified, ruff passes,
    no hardcoding, specialization conditions verified on 3+3 inputs.
    """
    blockers = []
    suggestions = []

    if testeval is None:
        blockers.append("No test results — run pytest")
    elif testeval.get("failed", 0) > 0:
        blockers.append(f"pytest has {testeval['failed']} failures — fix before proceeding")

    fallback_verified = context.get("fallback_verified", False) if context else False
    if not fallback_verified:
        blockers.append("Fallback path not manually verified")

    ruff_passes = context.get("ruff_passes", False) if context else False
    if not ruff_passes:
        suggestions.append("ruff check not confirmed passing")

    spec_verified = context.get("spec_condition_verified", False) if context else False
    if not spec_verified:
        suggestions.append(
            "Specialization conditions not verified on 3 hit + 3 fallback inputs"
        )

    # Cache check
    cache_cleared = context.get("cache_cleared", False) if context else False
    if not cache_cleared:
        suggestions.append(
            "Compilation cache may not have been cleared — "
            "old generated code might be served: rm -rf ~/.ninetoothed/"
        )

    return (len(blockers) == 0, blockers, suggestions)


def checkpoint_phase_5(
    testeval: dict | None = None,
    benchmark: dict | None = None,
    pitfall_stats: dict | None = None,
    context: dict | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Phase 5: Testing.

    Checks: >=6 new tests pass, existing tests still pass,
    adversarial tests pass, ntops tests pass, no false hits.
    """
    blockers = []
    suggestions = []

    new_tests = context.get("new_tests", {}) if context else {}
    hit_tests = new_tests.get("hit_tests", 0)
    fallback_tests = new_tests.get("fallback_tests", 0)
    structure_tests = new_tests.get("structure_tests", 0)

    if hit_tests < 2:
        blockers.append(f"Only {hit_tests}/2 specialization hit tests")
    if fallback_tests < 2:
        blockers.append(f"Only {fallback_tests}/2 fallback correctness tests")
    if structure_tests < 2:
        blockers.append(f"Only {structure_tests}/2 generated source structure tests")

    if testeval and testeval.get("failed", 0) > 0:
        blockers.append(f"pytest has {testeval['failed']} failures")

    adversarial_passed = context.get("adversarial_passed", False) if context else False
    if not adversarial_passed:
        suggestions.append("Adversarial edge cases not fully verified")

    false_hits = context.get("false_hits", -1) if context else -1
    if false_hits > 0:
        blockers.append(
            f"{false_hits} false specialization hits detected — "
            "tighten enable conditions"
        )

    ntops_ok = context.get("ntops_ok", False) if context else False
    if not ntops_ok:
        suggestions.append("ntops full test suite not confirmed passing")

    return (len(blockers) == 0, blockers, suggestions)


def checkpoint_phase_6(
    testeval: dict | None = None,
    benchmark: dict | None = None,
    pitfall_stats: dict | None = None,
    context: dict | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Phase 6: Benchmark.

    Checks: >=2 hit + >=2 fallback benchmarks collected,
    no speedup < 0.95 regression, reduction metrics recorded.
    """
    blockers = []
    suggestions = []

    if benchmark is None:
        blockers.append("No benchmark results collected")
    else:
        hit_count = benchmark.get("spec_hit_count", 0)
        fallback_count = benchmark.get("fallback_count", 0)

        if hit_count < 2:
            blockers.append(f"Only {hit_count}/2 hit case benchmarks")
        if fallback_count < 2:
            blockers.append(f"Only {fallback_count}/2 fallback case benchmarks")

        # Check for regressions
        regressions = benchmark.get("regressions", [])
        if regressions:
            blockers.append(
                f"Performance regressions detected (speedup < 0.95): {regressions}"
            )

        avg_speedup = benchmark.get("avg_speedup", 0)
        if avg_speedup < 1.0:
            suggestions.append(
                f"Average speedup {avg_speedup:.3f} < 1.0 — "
                "consider additional optimization"
            )

        avg_reduction = benchmark.get("avg_reduction", 0)
        if avg_reduction <= 0:
            suggestions.append(
                f"Average code reduction {avg_reduction:.3f} <= 0 — "
                "specialization may not be improving code"
            )

    return (len(blockers) == 0, blockers, suggestions)


def checkpoint_phase_7(
    testeval: dict | None = None,
    benchmark: dict | None = None,
    pitfall_stats: dict | None = None,
    context: dict | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Phase 7: Report and compliance.

    Checks: report PDF with 6 sections, HONOR_CODE.md signed,
    REFERENCE.md complete, PR description ready.
    """
    blockers = []
    suggestions = []

    report_ready = context.get("report_ready", False) if context else False
    if not report_ready:
        blockers.append("Competition report PDF not ready (6 required sections)")

    honor_code_ready = context.get("honor_code_ready", False) if context else False
    if not honor_code_ready:
        blockers.append("HONOR_CODE.md not signed and complete")

    reference_ready = context.get("reference_ready", False) if context else False
    if not reference_ready:
        blockers.append("REFERENCE.md not complete")

    pr_description_ready = context.get("pr_description_ready", False) if context else False
    if not pr_description_ready:
        blockers.append("PR description not prepared (7 required items)")

    return (len(blockers) == 0, blockers, suggestions)


def checkpoint_phase_8(
    testeval: dict | None = None,
    benchmark: dict | None = None,
    pitfall_stats: dict | None = None,
    context: dict | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Phase 8: Submission and cross-validation.

    Checks: final pytest all pass, ruff passes, PR content complete,
    ntops compatible, branch name and commit message compliant.
    """
    blockers = []
    suggestions = []

    if testeval and testeval.get("failed", 0) > 0:
        blockers.append(
            f"Final pytest has {testeval['failed']} failures — must be 0"
        )

    ruff_passes = context.get("ruff_passes", False) if context else False
    if not ruff_passes:
        blockers.append("ruff format/check not passing")

    pr_ready = context.get("pr_ready", False) if context else False
    if not pr_ready:
        blockers.append("PR not ready for submission")

    ntops_ok = context.get("ntops_ok", False) if context else False
    if not ntops_ok:
        suggestions.append(
            "ntops compatibility not confirmed — need final validation"
        )

    branch_ok = context.get("branch_name_ok", False) if context else False
    if not branch_ok:
        blockers.append("Branch name does not match required pattern")

    return (len(blockers) == 0, blockers, suggestions)


# Registry of all checkpoint functions
CHECKPOINTS: dict[str, CheckpointFn] = {
    "phase_0_setup": checkpoint_phase_0,
    "phase_1_environment": checkpoint_phase_1,
    "phase_2_weakness_analysis": checkpoint_phase_2,
    "phase_3_design": checkpoint_phase_3,
    "phase_4_implement": checkpoint_phase_4,
    "phase_5_test": checkpoint_phase_5,
    "phase_6_benchmark": checkpoint_phase_6,
    "phase_7_report": checkpoint_phase_7,
    "phase_8_submit": checkpoint_phase_8,
}


def evaluate_checkpoint(
    state_key: str,
    testeval: dict | None = None,
    benchmark: dict | None = None,
    pitfall_stats: dict | None = None,
    context: dict | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Evaluate the checkpoint for a given state.

    Args:
        state_key: The LoopState value string (e.g., "phase_4_implement").
        testeval: Results from TestEval.evaluate().
        benchmark: Results from BenchmarkRunner.
        pitfall_stats: Statistics from PitfallTracker.stats().
        context: Additional phase-specific context.

    Returns:
        (passed: bool, blockers: list[str], suggestions: list[str])
    """
    if state_key not in CHECKPOINTS:
        return (True, [], [f"No checkpoint defined for {state_key}"])

    return CHECKPOINTS[state_key](testeval, benchmark, pitfall_stats, context)
