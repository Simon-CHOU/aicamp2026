"""Tests for the loop engineering engine.

Verifies that the engine initializes correctly, state transitions work,
checkpoints evaluate properly, and the orchestrator produces valid output.
"""

import json
import pathlib
import tempfile
import pytest

from loop.state import (
    LoopState,
    LoopConfig,
    PhaseResult,
    LoopResult,
)
from loop.checkpoint import (
    CHECKPOINTS,
    evaluate_checkpoint,
    checkpoint_phase_0,
    checkpoint_phase_1,
    checkpoint_phase_4,
    checkpoint_phase_5,
)
from loop.engine import LoopEngine, create_engine
from loop.orchestrator import LoopOrchestrator


class TestLoopState:
    """Tests for LoopState enum."""

    def test_from_phase_number(self):
        assert LoopState.from_phase_number(0) == LoopState.SETUP
        assert LoopState.from_phase_number(4) == LoopState.IMPLEMENT
        assert LoopState.from_phase_number(8) == LoopState.SUBMIT
        assert LoopState.from_phase_number(99) == LoopState.BLOCKED

    def test_phase_number_property(self):
        assert LoopState.SETUP.phase_number == 0
        assert LoopState.IMPLEMENT.phase_number == 4
        assert LoopState.COMPLETE.phase_number == 9
        assert LoopState.BLOCKED.phase_number == -1

    def test_display_name(self):
        assert "Phase 4" in LoopState.IMPLEMENT.display_name
        assert "Phase 0" in LoopState.SETUP.display_name
        assert "完成" in LoopState.COMPLETE.display_name


class TestLoopConfig:
    """Tests for LoopConfig."""

    def test_defaults(self):
        config = LoopConfig(
            ninetoothed_path="/tmp/test",
            aicamp_path="/tmp/aicamp",
        )
        assert config.max_iterations_per_phase == 3
        assert config.correctness_threshold == 29
        assert config.speedup_target == 1.10
        assert config.reduction_target == 0.25
        assert config.current_state == LoopState.SETUP

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(pathlib.Path(tmpdir) / ".loop_state.json")
            config = LoopConfig(
                ninetoothed_path="/tmp/test",
                aicamp_path="/tmp/aicamp",
                state_file=state_file,
            )
            config.current_state = LoopState.IMPLEMENT
            config.iteration_count = 5
            config.save()

            loaded = LoopConfig.load(state_file)
            assert loaded.ninetoothed_path == "/tmp/test"
            assert loaded.current_state == LoopState.IMPLEMENT
            assert loaded.iteration_count == 5


class TestPhaseResult:
    """Tests for PhaseResult."""

    def test_to_from_dict(self):
        result = PhaseResult(
            phase_name="Phase 4: 实现",
            state=LoopState.IMPLEMENT,
            status="PASS",
            metrics={"speedup": 1.15},
            issues=[],
            next_action="Advance to Phase 5",
        )
        d = result.to_dict()
        restored = PhaseResult.from_dict(d)
        assert restored.phase_name == result.phase_name
        assert restored.state == result.state
        assert restored.status == result.status
        assert restored.metrics == result.metrics


class TestCheckpoints:
    """Tests for phase checkpoint functions."""

    def test_phase_0_pass(self):
        passed, blockers, _ = checkpoint_phase_0()
        # Will fail if packages aren't importable, but that's correct behavior
        # In test context we just verify the function runs without error
        assert isinstance(passed, bool)
        assert isinstance(blockers, list)

    def test_phase_1_no_testeval(self):
        passed, blockers, _ = checkpoint_phase_1()
        assert not passed
        assert any("No testeval" in b for b in blockers)

    def test_phase_1_with_passing_tests(self):
        testeval = {"passed": 42, "failed": 0, "skipped": 2, "known_wontfix": []}
        context = {"cuda_available": True, "triton_available": True}
        passed, blockers, _ = checkpoint_phase_1(
            testeval=testeval, context=context
        )
        assert passed

    def test_phase_4_with_failures(self):
        testeval = {"passed": 40, "failed": 2}
        passed, blockers, _ = checkpoint_phase_4(testeval=testeval)
        assert not passed
        assert any("failures" in b for b in blockers)

    def test_phase_5_insufficient_tests(self):
        context = {"new_tests": {"hit_tests": 1, "fallback_tests": 0, "structure_tests": 0}}
        passed, blockers, _ = checkpoint_phase_5(context=context)
        assert not passed
        assert len(blockers) >= 2  # missing fallback and structure tests

    def test_evaluate_checkpoint_unknown_state(self):
        passed, blockers, suggestions = evaluate_checkpoint("nonexistent")
        assert passed  # Unknown states pass by default
        assert len(suggestions) == 1

    def test_all_checkpoints_registered(self):
        """Verify all 9 phases have checkpoint functions."""
        for i in range(9):
            state = LoopState.from_phase_number(i)
            assert state.value in CHECKPOINTS, \
                f"Missing checkpoint for {state.value}"


class TestLoopOrchestrator:
    """Tests for LoopOrchestrator."""

    def test_initialization(self):
        config = LoopConfig(
            ninetoothed_path="/tmp/test",
            aicamp_path="/tmp/aicamp",
        )
        orchestrator = LoopOrchestrator(config)
        state = orchestrator.analyze_state()
        assert state["current_state"] == "phase_0_setup"
        assert state["phase_number"] == 0
        assert state["iteration_count"] == 0

    def test_generate_next_action(self):
        config = LoopConfig(
            ninetoothed_path="/tmp/test",
            aicamp_path="/tmp/aicamp",
        )
        orchestrator = LoopOrchestrator(config)
        action = orchestrator.generate_next_action()
        assert len(action) > 100  # Should be a substantial prompt
        assert "Phase 0" in action or "基础设施" in action

    def test_record_result_advances_iteration(self):
        config = LoopConfig(
            ninetoothed_path="/tmp/test",
            aicamp_path="/tmp/aicamp",
        )
        orchestrator = LoopOrchestrator(config)
        assert config.iteration_count == 0

        orchestrator.record_result(
            "test action",
            {"metrics": {"done": True}, "issues": []},
            status="PASS",
        )
        assert config.iteration_count == 1
        assert config.total_iterations == 1

    def test_advance_phase(self):
        config = LoopConfig(
            ninetoothed_path="/tmp/test",
            aicamp_path="/tmp/aicamp",
        )
        orchestrator = LoopOrchestrator(config)
        assert config.current_state == LoopState.SETUP

        new_state = orchestrator.advance_phase()
        assert new_state == LoopState.ENVIRONMENT
        assert config.iteration_count == 0  # Reset

    def test_advance_from_phase_8_goes_to_complete(self):
        config = LoopConfig(
            ninetoothed_path="/tmp/test",
            aicamp_path="/tmp/aicamp",
        )
        config.current_state = LoopState.SUBMIT
        orchestrator = LoopOrchestrator(config)

        new_state = orchestrator.advance_phase()
        assert new_state == LoopState.COMPLETE


class TestLoopEngine:
    """Tests for LoopEngine."""

    def test_create_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LoopEngine(LoopConfig(
                ninetoothed_path="/tmp/test",
                aicamp_path=tmpdir,
            ))
            engine._start_fresh()
            assert engine.config.current_state == LoopState.SETUP

    def test_get_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LoopEngine(LoopConfig(
                ninetoothed_path="/tmp/test",
                aicamp_path=tmpdir,
            ))
            engine._start_fresh()
            status = engine.get_status()
            assert status["current_state"] == "phase_0_setup"
            assert status["phase_number"] == 0
            assert "checkpoint_passed" in status
            assert "blockers" in status

    def test_is_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LoopEngine(LoopConfig(
                ninetoothed_path="/tmp/test",
                aicamp_path=tmpdir,
            ))
            engine._start_fresh()
            assert not engine.is_complete()

            engine.config.current_state = LoopState.COMPLETE
            assert engine.is_complete()

            engine.config.current_state = LoopState.BLOCKED
            assert engine.is_complete()

    def test_state_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(pathlib.Path(tmpdir) / ".loop_state.json")

            engine = LoopEngine(LoopConfig(
                ninetoothed_path="/tmp/nine",
                aicamp_path=tmpdir,
                state_file=state_file,
            ))
            engine._start_fresh()
            engine.config.current_state = LoopState.IMPLEMENT
            engine.config.iteration_count = 3
            engine.config.save()

            # Create new engine that loads the state
            engine2 = LoopEngine(LoopConfig(
                ninetoothed_path="/tmp/nine",
                aicamp_path=tmpdir,
                state_file=state_file,
            ))
            engine2.initialize()
            assert engine2.config.current_state == LoopState.IMPLEMENT
            assert engine2.config.iteration_count == 3


class TestLoopResult:
    """Tests for LoopResult."""

    def test_record_phase(self):
        result = LoopResult()
        pr = PhaseResult(
            phase_name="Phase 1",
            state=LoopState.ENVIRONMENT,
            status="PASS",
        )
        result.record_phase(pr)
        assert result.total_iterations == 1
        assert result.final_state == LoopState.ENVIRONMENT
        assert len(result.phase_results) == 1

    def test_to_dict(self):
        result = LoopResult()
        pr = PhaseResult(
            phase_name="Phase 4",
            state=LoopState.IMPLEMENT,
            status="PASS",
        )
        result.record_phase(pr)
        d = result.to_dict()
        assert d["total_iterations"] == 1
        assert len(d["phase_results"]) == 1
