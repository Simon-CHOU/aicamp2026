"""AI prompt templates for each phase of the T1-2-1 development loop.

Each template has slots like {current_state}, {testeval_result},
{benchmark_result}, {pitfall_issues}, {tasking_context} that get
filled by the LoopEngine before being sent to the AI.

These prompts guide the AI on what to analyze and what code to generate
at each phase of the loop.
"""

from typing import TypedDict


class PromptTemplate(TypedDict):
    """A prompt template with description and content."""

    description: str
    template: str


# Phase 0: Infrastructure setup
PROMPT_PHASE_0 = """
You are working on the NineToothed T1-2-1 compiler optimization project.
**Current Phase: {phase_display_name}**

## Tasking Context
{tasking_context}

## Current State
{current_state_summary}

## Your Task
Set up or verify the loop engineering infrastructure:
1. Confirm benchmark/ package is importable and its runner works
2. Confirm pitfall/ package can parse and append to .log files
3. Confirm loop/ engine initializes and tracks state correctly
4. Fix any issues found in the infrastructure

## Exit Criteria
{exit_criteria}

## Previous Pitfall Issues
{pitfall_issues}

## Action
1. Run the verification steps
2. Report what passes and what needs fixing
3. Fix any issues directly
"""

# Phase 1: Environment setup
PROMPT_PHASE_1 = """
You are working on the NineToothed T1-2-1 compiler optimization project.
**Current Phase: {phase_display_name}**

## Tasking Context
{tasking_context}

## Current State
- Test results: {testeval_result}
- CUDA available: {cuda_available}
- Triton available: {triton_available}
- Baseline metrics recorded: {baselines_recorded}

## Your Task
Set up and verify the development environment:
1. Confirm `pytest tests/` passes (note known WONTFIX issues)
2. Capture baseline generated Triton source for 2-3 typical operators
3. Record mask/stride/pointer counts as baseline metrics
4. Fix any environment issues (missing packages, CUDA, Triton version)

## Exit Criteria
{exit_criteria}

## Known Pitfall Issues
{pitfall_issues}

## Action
1. Run `pytest tests/ -x --timeout=300`
2. For any failures, determine if WONTFIX or need repair
3. Record baseline metrics to `benchmark/baselines/`
4. Report current state for checkpoint evaluation
"""

# Phase 2: Weakness analysis
PROMPT_PHASE_2 = """
You are working on the NineToothed T1-2-1 compiler optimization project.
**Current Phase: {phase_display_name}**

## Tasking Context
{tasking_context}

## Current State
- Test results: {testeval_result}
- Weakness cases analyzed: {weakness_cases_count}/2
- Prototype verified: {prototype_verified}
- Benchmark baselines: {benchmark_result}

## Your Task
Analyze the NineToothed code generation for weaknesses:

### Key Files to Study
1. `generation.py` — `_generate_pointers_and_mask()`, `_generate_overall_offsets_and_mask()`,
   `_generate_offsets_and_mask()`, `_generate_innermost_indices()`
2. `aot.py` — `_enumerate_variant_specs()`, `_build_variant()`
3. `tensor.py` — `Tensor.offsets()`, mask accumulation logic

### Target Scenarios
1. **element-wise (flatten + tile)**: stride still generated after flatten
2. **matmul with divisible block_size**: tail mask always generated
3. **scalar alpha parameter**: full pointer/stride/mask path for scalars
4. **conv2d constexpr stride/padding**: compile-time params not used

For each scenario, produce:
- Baseline source snippet (highlight inefficient parts)
- Quantified metrics: mask_expr_count, stride_expr_count, pointer_expr_count
- Theoretical optimal metrics after specialization
- Classification: redundant mask / stride / pointer / missed variant / broadcast

### Prototype Validation
Verify `is_contiguous` feasibility on symbolic tensors:
- Check if `stride[i] == prod(shape[i+1:])` can be proven symbolically
- If not feasible, document the fallback plan (divisible tile only)

## Exit Criteria
{exit_criteria}

## Previous Pitfall Issues
{pitfall_issues}

## Action
1. Read the key source files
2. Produce weakness analysis for >=2 cases with quantified metrics
3. Write to `docs/weakness_analysis.md`
4. Verify is_contiguous prototype
5. Record any new issues to pitfall log
"""

# Phase 3: Design
PROMPT_PHASE_3 = """
You are working on the NineToothed T1-2-1 compiler optimization project.
**Current Phase: {phase_display_name}**

## Tasking Context
{tasking_context}

## Weakness Analysis Results
{weakness_analysis_summary}

## Your Task
Design 1-2 specialization categories based on the weakness analysis.

### Available Categories
1. **Contiguous fast path** — simplify pointer expressions when contiguous
2. **Divisible tile fast path** — omit boundary mask when tile divides size
3. **Broadcast/scalar fast path** — simplify for broadcast dims and scalars
4. **Layout-known AOT variant** — leverage AOT info for better variants

### Design Requirements
For each selected category:
1. Write the **boolean predicate** that enables the specialization
2. Specify **which exact lines/sections** in generation.py/aot.py/tensor.py to modify
3. Document the **fallback path** (must go through original code)
4. Check: no hardcoded sizes, filenames, or benchmark names
5. Verify against scoring targets: speedup >= 1.10, reduction >= 0.25

### Priority Recommendation
- First: Divisible tile (tensor.py mask logic, lowest risk, clear metrics)
- Second: Contiguous path (generation.py stride elimination, higher complexity)

## Exit Criteria
{exit_criteria}

## Action
1. Select 1-2 categories
2. Write boolean predicates
3. Document design decisions to `docs/specialization_design.md`
4. Verify no hardcoding
5. Confirm scoring targets are achievable
"""

# Phase 4: Implementation
PROMPT_PHASE_4 = """
You are working on the NineToothed T1-2-1 compiler optimization project.
**Current Phase: {phase_display_name}**

## Design
{design_summary}

## Your Task
Implement the specialization(s) designed in Phase 3.

### Implementation Rules
1. **Preserve the original code path** — add if-else branches, never remove
2. **No test modifications** — don't delete, skip, or weaken existing tests
3. **Run pytest after every change** — catch regressions immediately
4. **Clear compilation cache after each change**: `rm -rf ~/.ninetoothed/`
5. **No hardcoding** — specialization conditions must be general

### For Divisible Tile Fast Path
- Modify `Tensor.offsets()` mask accumulation logic
- Add divisibility check before generating mask sub-conditions
- Reuse AOT divisibility info already available

### For Contiguous Fast Path
- Modify `_generate_overall_offsets_and_mask()` stride handling
- Add Tensor helper for symbolic contiguity check
- Only simplify when symbolically provable

### After Implementation
1. Run `pytest tests/ -x --timeout=300` — ALL must pass
2. Verify fallback: non-specializing inputs produce identical code to baseline
3. Verify hit: specializing inputs show reduced mask/stride/pointer counts
4. Run `ruff format --check . && ruff check .`
5. Manual check: 3 hit inputs + 3 fallback inputs behave correctly

## Exit Criteria
{exit_criteria}

## Code Context
- Source at: {ninetoothed_path}/src/ninetoothed/
- Tests at: {ninetoothed_path}/tests/

## Action
Implement the specialization(s) following the rules above. Report results after each step.
"""

# Phase 5: Testing
PROMPT_PHASE_5 = """
You are working on the NineToothed T1-2-1 compiler optimization project.
**Current Phase: {phase_display_name}**

## Implementation Summary
{implementation_summary}

## Your Task
Create comprehensive tests for the implemented specializations.

### Required New Tests (>=6 total)
1. **>=2 specialization hit tests**: verify specialization triggers on intended inputs
2. **>=2 fallback correctness tests**: verify non-specializing inputs still work
3. **>=2 generated source structure tests**: check mask/stride/pointer in output

### Adversarial Edge Cases (>=5 per specialization)
- Divisibility boundaries (e.g., size=17, tile=16)
- Mixed 1D/2D/3D dimensions
- Size-1 or empty dimensions
- Different dtype combinations (float32, float16, int32)
- Extreme sizes (very large, very small)

### Anti-False-Hit Verification
- Construct inputs that are CLOSE to but don't meet specialization conditions
- Verify they do NOT trigger the specialization path
- Each false hit costs 1 point in hidden evaluation

### Also Required
- Run ntops full test suite
- Run pytest with adversarial cases
- Record any new bugs to pitfall log

## Exit Criteria
{exit_criteria}

## Action
1. Write and run new tests
2. Design and test adversarial cases
3. Verify no false hits
4. Run ntops tests
5. Report results
"""

# Phase 6: Benchmark
PROMPT_PHASE_6 = """
You are working on the NineToothed T1-2-1 compiler optimization project.
**Current Phase: {phase_display_name}**

## Your Task
Collect comprehensive benchmark data for the implemented specializations.

### Required Data Per Case
- baseline_runtime_ms, submitted_runtime_ms, speedup
- specialization_hit (bool)
- Code metrics: mask_expr_count, stride_expr_count, pointer_expr_count
- source_line_count, variant_name

### Targets
- speedup >= 1.10 for full marks (1.00-1.10 linear scoring)
- reduction >= 0.25 for full marks (0-0.25 linear scoring)
- speedup < 0.95 = regression (0 points)

### Cases Required
- >=2 specialization hit cases
- >=2 fallback cases
- Output as JSON (per schema)

### If Issues Found
- speedup < 1.0 or reduction <= 0 → analyze root cause → go back to Phase 4
- Record all findings

## Exit Criteria
{exit_criteria}

## Current Benchmark Data
{benchmark_result}

## Action
1. Use benchmark/runner.py to collect all metrics
2. Export results as JSON
3. Analyze for regressions
4. Report summary
"""

# Phase 7: Report
PROMPT_PHASE_7 = """
You are working on the NineToothed T1-2-1 compiler optimization project.
**Current Phase: {phase_display_name}**

## Your Task
Prepare all submission materials.

### 1. Competition Report (PDF)
Six required sections:
1. Feature overview and change scope
2. Technical approach, core design, key code paths (including correctness proof)
3. Correctness verification methods and results
4. Metrics, test matrix, comparison data
5. Performance regressions, failed cases, unsupported scenarios
6. References, third-party tools, AI assistance disclosure

### 2. HONOR_CODE.md
- Independent work declaration
- AI assistance disclosure
- External code references
- Signature and date

### 3. REFERENCE.md
- All references (papers, docs, blogs)
- Reference implementations
- External tools

### 4. PR Description
Seven required items (see tasking.md Phase 7.4)

## Exit Criteria
{exit_criteria}

## Action
Prepare all submission materials.
"""

# Phase 8: Submission
PROMPT_PHASE_8 = """
You are working on the NineToothed T1-2-1 compiler optimization project.
**Current Phase: {phase_display_name}**

## Your Task
Final checks and PR submission.

### Final Checklist
- [ ] pytest tests/ all pass (no skips, no weakens)
- [ ] ruff format --check . && ruff check . pass
- [ ] No hardcoded sizes, filenames, benchmark names
- [ ] No undeclared external dependencies
- [ ] All new code has test coverage
- [ ] HONOR_CODE.md, REFERENCE.md, report PDF included
- [ ] PR description complete (7 items)
- [ ] Branch name: 2026-spring-Simon-CHOU-T1-2-1
- [ ] PR title: [2026春季][T1-2-1] Simon-CHOU
- [ ] PR target: InfiniTensor/ninetoothed main
- [ ] ntops verification complete (no separate PR needed)

### PR Info
- Fork: https://github.com/Simon-CHOU/ninetoothed
- Branch: 2026-spring-Simon-CHOU-T1-2-1
- Target: https://github.com/InfiniTensor/ninetoothed (main)

## Exit Criteria
{exit_criteria}

## Action
1. Run final checks
2. Fix any issues
3. Confirm PR readiness
"""

# Map phase state keys to their prompt templates
PROMPTS: dict[str, str] = {
    "phase_0_setup": PROMPT_PHASE_0,
    "phase_1_environment": PROMPT_PHASE_1,
    "phase_2_weakness_analysis": PROMPT_PHASE_2,
    "phase_3_design": PROMPT_PHASE_3,
    "phase_4_implement": PROMPT_PHASE_4,
    "phase_5_test": PROMPT_PHASE_5,
    "phase_6_benchmark": PROMPT_PHASE_6,
    "phase_7_report": PROMPT_PHASE_7,
    "phase_8_submit": PROMPT_PHASE_8,
}
