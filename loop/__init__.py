"""Loop engineering architecture for AI-driven continuous iteration.

Provides the engine that drives the T1-2-1 development loop:
testeval -> analyze -> implement -> benchmark -> repeat.

Key components:
- LoopEngine: main driver that orchestrates phases
- LoopOrchestrator: AI-facing interface for state analysis and next-action generation
- LoopState: state machine tracking which phase we're in
- PhaseResult: structured output from each phase execution
- Checkpoints: exit criteria for each phase

Usage:
    from loop.engine import LoopEngine
    engine = LoopEngine(config)
    engine.run_phase("phase_2_weakness_analysis")
    context = engine.get_context()  # For AI consumption
"""

from loop.state import LoopState, LoopConfig, PhaseResult, LoopResult
from loop.engine import LoopEngine
from loop.orchestrator import LoopOrchestrator
from loop.checkpoint import CHECKPOINTS, evaluate_checkpoint

__all__ = [
    "LoopState",
    "LoopConfig",
    "PhaseResult",
    "LoopResult",
    "LoopEngine",
    "LoopOrchestrator",
    "CHECKPOINTS",
    "evaluate_checkpoint",
]
