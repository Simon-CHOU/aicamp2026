"""LoopOrchestrator: AI-facing interface for state analysis and next-action generation.

The orchestrator is the "brain" that:
1. Reads current state from the LoopEngine
2. Analyzes testeval + benchmark + pitfall data
3. Generates the next action for the AI to take
4. Records results of AI actions
5. Evaluates whether phase exit criteria are met

This is the PRIMARY interface an AI agent uses to navigate the loop.
"""

import json
import pathlib
from typing import Optional
from datetime import datetime

from loop.state import LoopConfig, LoopState, PhaseResult, LoopResult
from loop.checkpoint import evaluate_checkpoint
from loop.prompts import PROMPTS


class LoopOrchestrator:
    """Orchestrates AI-driven iteration across T1-2-1 development phases.

    Usage:
        orchestrator = LoopOrchestrator(engine)
        context = orchestrator.analyze_state()
        next_action = orchestrator.generate_next_action()
        # AI executes the action...
        orchestrator.record_result(action, result)
        passed, blockers, suggestions = orchestrator.checkpoint()
    """

    def __init__(self, config: LoopConfig):
        """Initialize orchestrator with loop configuration.

        Args:
            config: LoopConfig with paths, thresholds, and current state.
        """
        self.config = config
        self._results: list[PhaseResult] = []
        self._action_log: list[dict] = []

    def analyze_state(self) -> dict:
        """Analyze the current state and return structured context for AI.

        Reads testeval results, benchmark data, and pitfall stats
        to provide a complete picture of the current situation.

        Returns:
            Dict with keys: current_state, phase_display_name,
            testeval_result, benchmark_result, pitfall_stats,
            exit_criteria_summary, iteration_count, max_iterations.
        """
        state_key = self.config.current_state.value

        # Try to load testeval results
        testeval = self._load_testeval()

        # Try to load benchmark results
        benchmark = self._load_benchmark()

        # Try to load pitfall stats
        pitfall_stats = self._load_pitfall_stats()

        # Get exit criteria for current phase
        _, blockers, suggestions = evaluate_checkpoint(
            state_key, testeval, benchmark, pitfall_stats
        )

        return {
            "current_state": state_key,
            "phase_display_name": self.config.current_state.display_name,
            "phase_number": self.config.current_state.phase_number,
            "testeval_result": testeval,
            "benchmark_result": benchmark,
            "pitfall_stats": pitfall_stats,
            "exit_criteria_summary": {
                "blockers": blockers,
                "suggestions": suggestions,
            },
            "iteration_count": self.config.iteration_count,
            "max_iterations": self.config.max_iterations_per_phase,
            "ninetoothed_path": self.config.ninetoothed_path,
            "aicamp_path": self.config.aicamp_path,
        }

    def generate_next_action(self) -> str:
        """Generate the next action the AI should take.

        Uses the current state and prompt template to produce
        a detailed action description with specific instructions.

        Returns:
            A prompt string the AI should follow.
        """
        state_key = self.config.current_state.value
        context = self.analyze_state()

        # Get the prompt template for this phase
        template = PROMPTS.get(state_key, "No prompt defined for phase {state_key}")

        # Fill in template slots
        action = template.format(
            phase_display_name=context["phase_display_name"],
            tasking_context=self._get_tasking_context(),
            current_state_summary=json.dumps(context, indent=2, ensure_ascii=False),
            testeval_result=json.dumps(context["testeval_result"], indent=2),
            cuda_available=context.get("cuda_available", "unknown"),
            triton_available=context.get("triton_available", "unknown"),
            baselines_recorded=context.get("baselines_recorded", "unknown"),
            weakness_cases_count=context.get("weakness_cases", 0),
            prototype_verified=context.get("prototype_verified", "unknown"),
            weakness_analysis_summary="See docs/weakness_analysis.md",
            design_summary="See docs/specialization_design.md",
            implementation_summary="Review git diff on current branch",
            benchmark_result=json.dumps(context["benchmark_result"], indent=2),
            pitfall_issues=self._get_pitfall_summary(),
            exit_criteria=json.dumps(
                context["exit_criteria_summary"], indent=2, ensure_ascii=False
            ),
            ninetoothed_path=self.config.ninetoothed_path,
            state_key=state_key,
        )

        return action

    def record_result(
        self,
        action_description: str,
        result: dict,
        status: str = "PASS",
    ) -> PhaseResult:
        """Record the result of an AI action.

        Args:
            action_description: What the AI attempted.
            result: Structured result of the action.
            status: "PASS", "FAIL", or "RETRY".

        Returns:
            PhaseResult recording this iteration.
        """
        phase_result = PhaseResult(
            phase_name=self.config.current_state.display_name,
            state=self.config.current_state,
            status=status,
            metrics=result.get("metrics", {}),
            issues=result.get("issues", []),
            next_action=result.get("next_action", ""),
            checkpoint_results=result.get("checkpoint_results", {}),
        )

        self._results.append(phase_result)
        self._action_log.append(
            {
                "action": action_description,
                "result": result,
                "status": status,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.config.iteration_count += 1
        self.config.total_iterations += 1

        return phase_result

    def checkpoint(self) -> tuple[bool, list[str], list[str]]:
        """Evaluate whether the current phase exit criteria are met.

        Returns:
            (passed: bool, blockers: list[str], suggestions: list[str])
            If passed is True, the loop engine should advance to the next phase.
        """
        state_key = self.config.current_state.value
        testeval = self._load_testeval()
        benchmark = self._load_benchmark()
        pitfall_stats = self._load_pitfall_stats()

        # Build context from accumulated results
        context = self._build_context()

        return evaluate_checkpoint(
            state_key, testeval, benchmark, pitfall_stats, context
        )

    def should_advance(self) -> tuple[bool, str]:
        """Determine if the loop should advance to the next phase.

        Returns:
            (should_advance: bool, reason: str)
        """
        passed, blockers, suggestions = self.checkpoint()

        if passed:
            return (True, "All exit criteria met")

        if self.config.iteration_count >= self.config.max_iterations_per_phase:
            return (
                True,
                f"Max iterations ({self.config.max_iterations_per_phase}) "
                f"reached. Advancing with unresolved issues: {blockers}",
            )

        reason = f"Blocked: {', '.join(blockers)}" if blockers else \
                 f"Suggestions: {', '.join(suggestions)}"
        return (False, reason)

    def advance_phase(self) -> LoopState:
        """Advance to the next phase.

        Resets iteration count and moves the state machine forward.

        Returns:
            The new LoopState.
        """
        current_num = self.config.current_state.phase_number
        if current_num < 8:
            new_state = LoopState.from_phase_number(current_num + 1)
        elif current_num == 8:
            new_state = LoopState.COMPLETE
        else:
            new_state = LoopState.COMPLETE

        self.config.current_state = new_state
        self.config.iteration_count = 0
        self.config.save()

        return new_state

    def get_context_for_ai(self) -> str:
        """Get a complete, AI-ready context string.

        This is the MAIN entry point for AI agents. It provides
        everything the AI needs to understand the current state
        and take the next action.

        Returns:
            A formatted string with all context needed by the AI.
        """
        state = self.analyze_state()
        next_action = self.generate_next_action()

        lines = [
            "=" * 70,
            f"LOOP ENGINE STATE: {state['phase_display_name']}",
            f"Iteration: {state['iteration_count']}/{state['max_iterations']}",
            "=" * 70,
            "",
            "## Current Status",
            f"State: {state['current_state']}",
            f"Phase: {state['phase_number']}",
            "",
            "## Exit Criteria",
        ]

        for b in state["exit_criteria_summary"].get("blockers", []):
            lines.append(f"  🔴 BLOCKER: {b}")
        for s in state["exit_criteria_summary"].get("suggestions", []):
            lines.append(f"  🟡 SUGGESTION: {s}")

        if state["testeval_result"]:
            lines.append("")
            lines.append("## Test Results")
            lines.append(
                f"  Passed: {state['testeval_result'].get('passed', 'N/A')}"
            )
            lines.append(
                f"  Failed: {state['testeval_result'].get('failed', 'N/A')}"
            )

        if state["benchmark_result"]:
            lines.append("")
            lines.append("## Benchmark Results")
            lines.append(json.dumps(state["benchmark_result"], indent=2))

        lines.append("")
        lines.append("## Next Action")
        lines.append(next_action)

        return "\n".join(lines)

    # --- Private helpers ---

    def _load_testeval(self) -> dict | None:
        """Load latest testeval results if available."""
        result_path = pathlib.Path(self.config.aicamp_path) / "results" / "latest_testeval.json"
        if result_path.exists():
            try:
                with open(result_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _load_benchmark(self) -> dict | None:
        """Load latest benchmark results if available."""
        result_path = pathlib.Path(self.config.aicamp_path) / "results" / "latest_benchmark.json"
        if result_path.exists():
            try:
                with open(result_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _load_pitfall_stats(self) -> dict | None:
        """Load pitfall statistics."""
        try:
            from pitfall.tracker import PitfallTracker

            tracker = PitfallTracker(
                log_dir=str(
                    pathlib.Path(self.config.aicamp_path) / "docs"
                )
            )
            return tracker.stats()
        except Exception:
            return None

    def _get_pitfall_summary(self) -> str:
        """Get a summary of pitfall issues."""
        try:
            from pitfall.tracker import PitfallTracker

            tracker = PitfallTracker(
                log_dir=str(
                    pathlib.Path(self.config.aicamp_path) / "docs"
                )
            )
            unresolved = tracker.get_unresolved()
            if unresolved:
                lines = ["## Unresolved Pitfall Issues"]
                for p in unresolved:
                    lines.append(f"- #{p.number}: {p.description} [{p.category.value}]")
                return "\n".join(lines)
            return "No unresolved pitfall issues."
        except Exception:
            return "Unable to load pitfall data."

    def _get_tasking_context(self) -> str:
        """Load the tasking.md content for context."""
        tasking_path = pathlib.Path(self.config.aicamp_path) / "docs" / "tasking.md"
        if tasking_path.exists():
            with open(tasking_path) as f:
                content = f.read()
            # Return the section relevant to the current phase
            phase_num = self.config.current_state.phase_number
            phase_marker = f"Phase {phase_num}:"
            if phase_marker in content:
                start = content.find(phase_marker)
                # Find the next phase marker or end
                next_phase = f"Phase {phase_num + 1}:"
                next_idx = content.find(next_phase, start)
                if next_idx > start:
                    return content[start:next_idx].strip()
                return content[start:].strip()
            return content[:3000]  # Fallback: first 3000 chars
        return "tasking.md not found"

    def _build_context(self) -> dict:
        """Build context dict from accumulated results for checkpoint evaluation."""
        context = {}

        # Extract metrics from the last phase result
        if self._results:
            last = self._results[-1]
            context.update(last.metrics)
            context["last_status"] = last.status
            context["last_issues"] = last.issues

        context["iteration_count"] = self.config.iteration_count
        context["max_iterations"] = self.config.max_iterations_per_phase
        context["ninetoothed_path"] = self.config.ninetoothed_path

        return context
