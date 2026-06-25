# Loop Engineering Architecture

AI 驱动的持续迭代引擎，用于 NineToothed T1-2-1 编译优化项目。

## 概述

Loop Engine 是一个状态机驱动的开发循环系统，它将 T1-2-1 的 8 个 Phase 串联起来，
为 AI agent 提供结构化的上下文、明确的出口条件、以及自动化的状态推进。

### 核心循环

```
testeval → analyze → implement → benchmark → repeat
    ↑                                          │
    └──────────── pitfall tracking ────────────┘
```

## 架构

```
loop/
├── __init__.py        # Package exports
├── engine.py          # LoopEngine: 主驱动器
├── state.py           # 状态机: LoopState, LoopConfig, PhaseResult
├── prompts.py         # 每阶段 AI prompt 模板
├── orchestrator.py    # LoopOrchestrator: AI 接口
└── checkpoint.py      # 每阶段出口条件（布尔谓词）
```

### 组件职责

| 组件 | 职责 |
|------|------|
| `LoopEngine` | 初始化、状态管理、phase 推进、持久化 |
| `LoopOrchestrator` | AI 接口: 分析状态、生成 prompt、记录结果、评估 checkpoint |
| `LoopConfig` | 配置: 路径、阈值、最大迭代次数 |
| `LoopState` | 状态枚举: 9 个 Phase + COMPLETE + BLOCKED |
| `PhaseResult` | 单次 Phase 执行的结构化结果 |
| `CheckpointFn` | Phase 出口条件的布尔谓词函数 |

## 快速开始

### 1. 初始化

```bash
cd /home/simon/ninetooth2026/aicamp2026
python -m loop.engine --init
```

### 2. 查看状态

```bash
python -m loop.engine --status
```

输出:
```
============================================================
  LOOP ENGINE: Phase 2: Weakness Analysis
============================================================
  Phase:         2/8
  Iteration:     1/3
  Total Iters:   5
  Checkpoint:    ❌ NOT MET

  🔴 Blockers:
     - Only 1/2 weakness cases analyzed
     - docs/weakness_analysis.md not found
============================================================
```

### 3. AI 获取上下文

```python
from loop.engine import create_engine

engine = create_engine(
    ninetoothed_path="/home/simon/ninetooth2026/ninetoothed",
    aicamp_path="/home/simon/ninetooth2026/aicamp2026",
)

# AI reads this to understand what to do next
context = engine.get_context()
print(context)
```

### 4. 检查是否可以推进

```bash
python -m loop.engine --checkpoint
```

### 5. 推进到下一 Phase

```bash
python -m loop.engine --advance
```

## AI 集成模式

AI agent 使用 Loop Engine 的标准模式:

```
1. engine = create_engine(...)
2. context = engine.orchestrator.get_context_for_ai()
3. AI 读取 context，理解当前状态和下一步任务
4. AI 执行任务（读代码、写代码、跑测试等）
5. AI 记录结果: orchestrator.record_result(action, result)
6. should_advance, reason = orchestrator.should_advance()
7. if should_advance: orchestrator.advance_phase()
8. 重复直到 COMPLETE
```

## Phase 概览

| Phase | 状态 | 出口条件关键项 |
|-------|------|---------------|
| 0 | 基础设施搭建 | benchmark/pitfall/loop 包可 import |
| 1 | 环境搭建 | pytest 通过, CUDA 可用, baselines 记录 |
| 2 | Weakness 分析 | >=2 cases 分析完成, 量化指标记录 |
| 3 | 设计选择 | 1-2 类别选定, 布尔谓词写定 |
| 4 | 实现 | pytest 全通过, fallback 验证 |
| 5 | 测试 | >=6 新测试, 对抗性测试通过 |
| 6 | Benchmark | >=4 cases 数据, speedup >= 1.0 |
| 7 | 报告合规 | 报告/Code/Reference 齐全 |
| 8 | 提交验证 | 最终检查, PR 就绪 |

## 状态持久化

状态自动保存到 `aicamp2026/.loop_state.json`:
```json
{
  "current_state": "phase_4_implement",
  "iteration_count": 2,
  "total_iterations": 15,
  "last_updated": "2026-06-28T14:30:00"
}
```

会话中断后重新初始化会自动恢复状态。

## 与其他包的关系

```
loop/engine.py ──读取──→ docs/tasking.md
       │
       ├──调用──→ benchmark/testeval.py  (TestEval)
       ├──调用──→ benchmark/runner.py    (BenchmarkRunner)
       └──调用──→ pitfall/tracker.py     (PitfallTracker)
```
