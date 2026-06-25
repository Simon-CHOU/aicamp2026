"""State machine and data models for the loop engineering engine.

Tracks which phase the project is in, phase exit criteria,
and accumulates results across iterations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
import pathlib
from datetime import datetime


class LoopState(Enum):
    """Current phase of the T1-2-1 development loop."""

    SETUP = "phase_0_setup"
    ENVIRONMENT = "phase_1_environment"
    WEAKNESS_ANALYSIS = "phase_2_weakness_analysis"
    DESIGN = "phase_3_design"
    IMPLEMENT = "phase_4_implement"
    TEST = "phase_5_test"
    BENCHMARK = "phase_6_benchmark"
    REPORT = "phase_7_report"
    SUBMIT = "phase_8_submit"
    COMPLETE = "complete"
    BLOCKED = "blocked"

    @classmethod
    def from_phase_number(cls, n: int) -> "LoopState":
        """Map phase number (0-8) to LoopState."""
        mapping = {
            0: cls.SETUP,
            1: cls.ENVIRONMENT,
            2: cls.WEAKNESS_ANALYSIS,
            3: cls.DESIGN,
            4: cls.IMPLEMENT,
            5: cls.TEST,
            6: cls.BENCHMARK,
            7: cls.REPORT,
            8: cls.SUBMIT,
        }
        return mapping.get(n, cls.BLOCKED)

    @property
    def phase_number(self) -> int:
        """Extract phase number from state."""
        mapping = {
            LoopState.SETUP: 0,
            LoopState.ENVIRONMENT: 1,
            LoopState.WEAKNESS_ANALYSIS: 2,
            LoopState.DESIGN: 3,
            LoopState.IMPLEMENT: 4,
            LoopState.TEST: 5,
            LoopState.BENCHMARK: 6,
            LoopState.REPORT: 7,
            LoopState.SUBMIT: 8,
            LoopState.COMPLETE: 9,
            LoopState.BLOCKED: -1,
        }
        return mapping.get(self, -1)

    @property
    def display_name(self) -> str:
        """Human-readable phase name."""
        names = {
            LoopState.SETUP: "Phase 0: Loop 工程基础设施搭建",
            LoopState.ENVIRONMENT: "Phase 1: 环境搭建与基线确认",
            LoopState.WEAKNESS_ANALYSIS: "Phase 2: Weakness Analysis",
            LoopState.DESIGN: "Phase 3: 选择特化类别与设计",
            LoopState.IMPLEMENT: "Phase 4: 实现",
            LoopState.TEST: "Phase 5: 测试",
            LoopState.BENCHMARK: "Phase 6: Benchmark",
            LoopState.REPORT: "Phase 7: 报告与合规",
            LoopState.SUBMIT: "Phase 8: 提交与交叉验证",
            LoopState.COMPLETE: "完成",
            LoopState.BLOCKED: "阻塞",
        }
        return names.get(self, str(self))


@dataclass
class LoopConfig:
    """Configuration for the loop engineering engine.

    Attributes:
        ninetoothed_path: Path to the ninetoothed repo.
        aicamp_path: Path to the aicamp2026 directory.
        ntops_path: Path to the ntops repo (for validation).
        max_iterations_per_phase: Maximum loop iterations before forcing progress.
        correctness_threshold: Minimum hidden correctness score (29/30).
        speedup_target: Target speedup for full marks (1.10).
        reduction_target: Target code metric reduction for full marks (0.25).
        state_file: Path to persist loop state.
        current_state: Current phase of the loop.
        iteration_count: Number of iterations in the current phase.
        total_iterations: Total iterations across all phases.
    """

    ninetoothed_path: str = ""
    aicamp_path: str = ""
    ntops_path: str = ""
    max_iterations_per_phase: int = 3
    correctness_threshold: int = 29  # out of 30
    speedup_target: float = 1.10
    reduction_target: float = 0.25
    state_file: str = ""
    current_state: LoopState = LoopState.SETUP
    iteration_count: int = 0
    total_iterations: int = 0

    def __post_init__(self):
        if not self.state_file:
            self.state_file = str(
                pathlib.Path(self.aicamp_path) / ".loop_state.json"
                if self.aicamp_path
                else ".loop_state.json"
            )

    def save(self) -> None:
        """Persist current config to state file."""
        data = {
            "ninetoothed_path": self.ninetoothed_path,
            "aicamp_path": self.aicamp_path,
            "ntops_path": self.ntops_path,
            "max_iterations_per_phase": self.max_iterations_per_phase,
            "correctness_threshold": self.correctness_threshold,
            "speedup_target": self.speedup_target,
            "reduction_target": self.reduction_target,
            "current_state": self.current_state.value,
            "iteration_count": self.iteration_count,
            "total_iterations": self.total_iterations,
            "last_updated": datetime.now().isoformat(),
        }
        pathlib.Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, state_file: str) -> "LoopConfig":
        """Load config from persisted state file."""
        with open(state_file) as f:
            data = json.load(f)
        config = cls(
            ninetoothed_path=data.get("ninetoothed_path", ""),
            aicamp_path=data.get("aicamp_path", ""),
            ntops_path=data.get("ntops_path", ""),
            max_iterations_per_phase=data.get("max_iterations_per_phase", 3),
            correctness_threshold=data.get("correctness_threshold", 29),
            speedup_target=data.get("speedup_target", 1.10),
            reduction_target=data.get("reduction_target", 0.25),
        )
        config.current_state = LoopState(data.get("current_state", "phase_0_setup"))
        config.iteration_count = data.get("iteration_count", 0)
        config.total_iterations = data.get("total_iterations", 0)
        config.state_file = state_file
        return config


@dataclass
class PhaseResult:
    """Result of executing one phase of the loop.

    Attributes:
        phase_name: Name of the phase that was executed.
        state: Current loop state after phase execution.
        status: "PASS", "FAIL", "RETRY", or "BLOCKED".
        metrics: Dict of phase-specific metrics.
        issues: List of issues found during phase execution.
        next_action: Description of what the AI should do next.
        checkpoint_results: Results of evaluating phase exit criteria.
        timestamp: When this result was recorded.
    """

    phase_name: str = ""
    state: LoopState = LoopState.SETUP
    status: str = "PASS"  # PASS, FAIL, RETRY, BLOCKED
    metrics: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    next_action: str = ""
    checkpoint_results: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "phase_name": self.phase_name,
            "state": self.state.value,
            "status": self.status,
            "metrics": self.metrics,
            "issues": self.issues,
            "next_action": self.next_action,
            "checkpoint_results": self.checkpoint_results,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhaseResult":
        return cls(
            phase_name=data.get("phase_name", ""),
            state=LoopState(data.get("state", "phase_0_setup")),
            status=data.get("status", "PASS"),
            metrics=data.get("metrics", {}),
            issues=data.get("issues", []),
            next_action=data.get("next_action", ""),
            checkpoint_results=data.get("checkpoint_results", {}),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class LoopResult:
    """Accumulated result of running the full loop.

    Attributes:
        phase_results: Results from each phase executed.
        total_iterations: Total iterations across all phases.
        final_state: The final loop state.
        summary: Human-readable summary of the entire run.
        started_at: When the loop was started.
        completed_at: When the loop completed (or was blocked).
    """

    phase_results: list[PhaseResult] = field(default_factory=list)
    total_iterations: int = 0
    final_state: LoopState = LoopState.SETUP
    summary: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""

    def record_phase(self, result: PhaseResult) -> None:
        """Record a phase result."""
        self.phase_results.append(result)
        self.total_iterations += 1
        self.final_state = result.state
        self.completed_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "phase_results": [r.to_dict() for r in self.phase_results],
            "total_iterations": self.total_iterations,
            "final_state": self.final_state.value,
            "summary": self.summary,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
