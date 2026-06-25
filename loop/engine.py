"""LoopEngine: main driver for the AI-driven development loop.

The LoopEngine is the primary entry point. It:
1. Initializes with a LoopConfig that defines paths and thresholds
2. Tracks which phase we're in and how many iterations we've done
3. Provides get_context() for AI consumption
4. Evaluates checkpoints to decide when to advance phases
5. Persists state so sessions can be resumed

Usage:
    from loop.engine import LoopEngine
    from loop.state import LoopConfig

    config = LoopConfig(
        ninetoothed_path="/home/simon/ninetooth2026/ninetoothed",
        aicamp_path="/home/simon/ninetooth2026/aicamp2026",
    )
    engine = LoopEngine(config)
    engine.initialize()

    # AI reads this to understand what to do:
    context = engine.get_context()

    # After AI does work, record result:
    engine.record_result(phase_result)

    # Check if we can move on:
    if engine.checkpoint_passed():
        engine.advance_phase()
"""

import json
import pathlib
import sys
from typing import Optional

from loop.state import LoopConfig, LoopState, PhaseResult, LoopResult
from loop.orchestrator import LoopOrchestrator


class LoopEngine:
    """Main driver for the T1-2-1 development loop.

    Manages phase transitions, checkpoint evaluation, state persistence,
    and provides the AI-facing interface via the orchestrator.

    Usage:
        engine = LoopEngine(config)
        engine.initialize()
        orchestrator = engine.orchestrator
        context = orchestrator.get_context_for_ai()
        # AI works based on context...
        orchestrator.record_result("did X", result)
        should_advance, reason = orchestrator.should_advance()
    """

    def __init__(self, config: LoopConfig):
        """Initialize the loop engine.

        Args:
            config: LoopConfig with ninetoothed_path, aicamp_path, etc.
        """
        self.config = config
        self.orchestrator = LoopOrchestrator(config)
        self._loop_result = LoopResult()

        # Ensure results directory exists
        results_dir = pathlib.Path(config.aicamp_path) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Initialize or resume the loop engine.

        If a state file exists, resume from it.
        Otherwise, start from Phase 0.
        """
        state_path = pathlib.Path(self.config.state_file)

        if state_path.exists():
            try:
                self.config = LoopConfig.load(str(state_path))
                self.orchestrator = LoopOrchestrator(self.config)
                print(f"Resumed loop from state file: {self.config.current_state.display_name}")
            except (json.JSONDecodeError, KeyError, OSError) as e:
                print(f"Warning: Could not load state file ({e}), starting fresh")
                self._start_fresh()
        else:
            self._start_fresh()

    def _start_fresh(self) -> None:
        """Start a fresh loop from Phase 0."""
        self.config.current_state = LoopState.SETUP
        self.config.iteration_count = 0
        self.config.total_iterations = 0
        self.config.save()

    def get_context(self) -> str:
        """Get the full AI-ready context for the current state.

        This is the primary method AI agents call to understand
        what to do next.

        Returns:
            Formatted string with current state, test/benchmark results,
            exit criteria, and next action prompt.
        """
        return self.orchestrator.get_context_for_ai()

    def get_status(self) -> dict:
        """Get a concise status summary.

        Returns:
            Dict with current_state, phase_number, iteration_count,
            max_iterations, checkpoint_status.
        """
        passed, blockers, suggestions = self.orchestrator.checkpoint()

        return {
            "current_state": self.config.current_state.value,
            "phase_display_name": self.config.current_state.display_name,
            "phase_number": self.config.current_state.phase_number,
            "iteration_count": self.config.iteration_count,
            "max_iterations": self.config.max_iterations_per_phase,
            "checkpoint_passed": passed,
            "blockers": blockers,
            "suggestions": suggestions,
            "total_iterations": self.config.total_iterations,
        }

    def checkpoint_passed(self) -> bool:
        """Check if current phase exit criteria are met.

        Returns:
            True if the phase is complete and we should advance.
        """
        passed, _, _ = self.orchestrator.checkpoint()
        return passed

    def advance_phase(self) -> LoopState:
        """Advance to the next phase. Call after checkpoint_passed() returns True.

        Returns:
            The new LoopState.
        """
        # Record the current phase result before advancing
        passed, blockers, suggestions = self.orchestrator.checkpoint()

        phase_result = PhaseResult(
            phase_name=self.config.current_state.display_name,
            state=self.config.current_state,
            status="PASS" if passed else "FAIL",
            metrics={"blockers": blockers, "suggestions": suggestions},
            issues=blockers,
            next_action="Advancing to next phase",
            checkpoint_results={
                "passed": passed,
                "blockers": blockers,
                "suggestions": suggestions,
            },
        )
        self._loop_result.record_phase(phase_result)

        # Advance
        new_state = self.orchestrator.advance_phase()
        self.config.save()

        return new_state

    def record_result(self, result: PhaseResult) -> None:
        """Record a phase result from AI action.

        Args:
            result: PhaseResult from the orchestrator.
        """
        self._loop_result.record_phase(result)
        self.config.save()

    def is_complete(self) -> bool:
        """Check if the entire loop is complete."""
        return self.config.current_state in (LoopState.COMPLETE, LoopState.BLOCKED)

    def print_status(self) -> None:
        """Print a human-readable status report to stdout."""
        status = self.get_status()

        print()
        print("=" * 60)
        print(f"  LOOP ENGINE: {status['phase_display_name']}")
        print("=" * 60)
        print(f"  Phase:         {status['phase_number']}/8")
        print(f"  Iteration:     {status['iteration_count']}/{status['max_iterations']}")
        print(f"  Total Iters:   {status['total_iterations']}")
        print(f"  Checkpoint:    {'✅ PASSED' if status['checkpoint_passed'] else '❌ NOT MET'}")
        print()

        if status["blockers"]:
            print("  🔴 Blockers:")
            for b in status["blockers"]:
                print(f"     - {b}")
            print()

        if status["suggestions"]:
            print("  🟡 Suggestions:")
            for s in status["suggestions"]:
                print(f"     - {s}")
            print()

        print("=" * 60)


def create_engine(
    ninetoothed_path: str,
    aicamp_path: str,
    ntops_path: str = "",
) -> LoopEngine:
    """Factory function to create and initialize a LoopEngine.

    Args:
        ninetoothed_path: Path to the ninetoothed repo.
        aicamp_path: Path to the aicamp2026 directory.
        ntops_path: Optional path to ntops repo for validation.

    Returns:
        Initialized LoopEngine ready for AI use.
    """
    config = LoopConfig(
        ninetoothed_path=ninetoothed_path,
        aicamp_path=aicamp_path,
        ntops_path=ntops_path,
    )
    engine = LoopEngine(config)
    engine.initialize()
    return engine


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Loop Engine for T1-2-1 AI-driven development"
    )
    parser.add_argument(
        "--init", action="store_true", help="Initialize loop engine state"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current loop status"
    )
    parser.add_argument(
        "--checkpoint", action="store_true", help="Evaluate current phase checkpoint"
    )
    parser.add_argument(
        "--advance", action="store_true", help="Advance to next phase"
    )
    parser.add_argument(
        "--context", action="store_true", help="Print full AI context"
    )
    parser.add_argument(
        "--ninetoothed-path",
        default="/home/simon/ninetooth2026/ninetoothed",
        help="Path to ninetoothed repo",
    )
    parser.add_argument(
        "--aicamp-path",
        default="/home/simon/ninetooth2026/aicamp2026",
        help="Path to aicamp2026 directory",
    )

    args = parser.parse_args()

    engine = create_engine(args.ninetoothed_path, args.aicamp_path)

    if args.init:
        engine._start_fresh()
        print("Loop engine initialized.")
        engine.print_status()

    if args.status:
        engine.print_status()

    if args.checkpoint:
        passed, blockers, suggestions = engine.orchestrator.checkpoint()
        if passed:
            print("✅ Checkpoint PASSED — ready to advance")
        else:
            print("❌ Checkpoint NOT MET")
            for b in blockers:
                print(f"  🔴 {b}")
            for s in suggestions:
                print(f"  🟡 {s}")

    if args.advance:
        if engine.checkpoint_passed():
            new_state = engine.advance_phase()
            print(f"Advanced to: {new_state.display_name}")
        else:
            print("Cannot advance: checkpoint not met. Use --force to override.")

    if args.context:
        print(engine.get_context())
