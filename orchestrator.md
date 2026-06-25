# T1-2-1 满分攻略：Loop Orchestrator 驱动方案

> **目标**: 利用已搭建的 loop orchestrator 架构，系统化达成赛题全部要求，冲击 100/100 满分。
>
> **核心思路**: 赛题评分标准的每一项都映射到 loop 的特定 Phase + Checkpoint，不存在"自由发挥"空间——每个 Phase 的出口条件就是评分标准的工程化表达。

---

## 一、赛题要求 → Loop Phase 映射总表

| 赛题要求 | 对应 Loop Phase | 对应 Checkpoint | 分值 |
|---------|----------------|-----------------|------|
| 固定任务 1: Weakness Analysis | Phase 2 | Checkpoint 2 | — |
| 固定任务 2: 实现特化 | Phase 3 + Phase 4 | Checkpoint 3 + 4 | — |
| 固定任务 3: 启用条件 + fallback | Phase 3 | Checkpoint 3 | — |
| 固定任务 4: 保持既有测试 | Phase 4 | Checkpoint 4 | — |
| 固定任务 5: 新增 source 结构测试 | Phase 5 | Checkpoint 5 | — |
| 固定任务 6: Benchmark | Phase 6 | Checkpoint 6 | — |
| 必须交付 1: 代码改动 | Phase 4 | Checkpoint 4 | — |
| 必须交付 2: 测试 (≥6 个) | Phase 5 | Checkpoint 5 | — |
| 必须交付 3: Benchmark 数据 | Phase 6 | Checkpoint 6 | — |
| 必须交付 4: 报告 | Phase 7 | Checkpoint 7 | — |
| Correctness (30 分) | Phase 5 | Checkpoint 5 | 30 |
| Specialization Coverage (20 分) | Phase 3 + 5 | Checkpoint 3 + 5 | 20 |
| Generated Code Metric (20 分) | Phase 6 | Checkpoint 6 | 20 |
| Runtime (20 分) | Phase 6 | Checkpoint 6 | 20 |
| 工程与报告质量 (10 分) | Phase 7 | Checkpoint 7 | 10 |

---

## 二、满分路线图：每个评分维度的致胜策略

### 2.1 Correctness — 30/30（门槛：≥29/30，否则总分上限 40）

**这是最重要的一维。若此维度不达标，满分直接不可能。**

#### Loop Orchestrator 的角色

```
Phase 5: Test
  ├── TestEval.run_pytest()          → 确保既有测试 0 失败
  ├── TestEval.check_specialization() → 验证特化命中/不命中
  ├── TestEval.check_fallback()       → 验证回退路径正确
  └── TestEval.evaluate()            → 综合评测 → 喂入 Checkpoint 5
```

#### 满分策略

| 策略 | 具体做法 | 对应 Checkpoint |
|------|---------|----------------|
| **对抗性场景覆盖** | 每特化 ≥5 个边缘 case：整除/非整除边界 (size=17 tile=16)、1D/2D/3D 混合、空维度、不同 dtype、极端尺寸 | Checkpoint 5: `adversarial_passed` |
| **防误命中验证** | 构造接近但不满足特化条件的输入，验证**绝不**错误命中特化。误命中在隐藏评测中每次扣 1 分 | Checkpoint 5: `false_hits == 0` |
| **数值等价性** | 特化路径生成的 kernel 输出与 baseline 做 `torch.allclose` 对比。特化只改 mask/pointer/stride，浮点路径不变 | 内建于 `TestEval.check_fallback()` |
| **ntops 兼容性** | 在 ntops 全量测试上验证 ninetoothed 改动不破坏任何上游算子 | Checkpoint 5: `ntops_ok` |
| **编译缓存清除** | 每次生成逻辑改动后执行 `rm -rf ~/.ninetoothed/`，确保测试的是新代码而非旧缓存 | Checkpoint 4: `cache_cleared` |

#### Loop 迭代策略

```
while not Checkpoint 5 passed:
    testeval = TestEval.evaluate()
    if testeval.correctness.failed > 0:
        → 分析失败原因 → 记录 pitfall
        → 回到 Phase 4 修复代码
        → 重新执行 Phase 4.4 自查
        → 重新跑 Phase 5
    if adversarial.passed < 5:
        → 补充边缘 case
    if false_hits > 0:
        → 收紧特化启用条件（宁可漏过也不错杀）
```

**关键洞察**: 隐藏测试 29/30 门槛意味着最多只能错 1 个用例。对抗性场景必须覆盖到"即使隐藏测试用例的设计者刻意构造了边界输入，我们也能正确处理"的程度。

---

### 2.2 Specialization Coverage — 20/20

**评分规则**: 12 个隐藏特化命中用例，每个正确命中得 1 分。每个应回退的用例错误命中扣 1 分。覆盖率按 `max(0, hit_score - false_hit_penalty) / 12` 折算到 20 分。

#### Loop Orchestrator 的角色

```
Phase 3: Design
  ├── 设计布尔谓词（启用条件）
  └── Checkpoint 3: predicates_written, no hardcoding

Phase 5: Test  
  ├── TestEval.check_specialization()
  └── Checkpoint 5: hit_tests >= 2, false_hits == 0
```

#### 满分策略

| 策略 | 具体做法 |
|------|---------|
| **谓词设计优先** | 先写布尔谓词，再写实现。谓词必须满足：(1) 基于符号信息可判定 (2) 对符合条件的所有输入返回 True (3) 对不符合条件的任何输入返回 False |
| **覆盖最大化** | 从 4 个特化类别中选择 2 个，用**不同维度**覆盖更多场景。Divisible tile + Contiguous 的组合天然覆盖了 element-wise、matmul、conv 的大部分输入 |
| **零误命中** | 这是扣分项，必须为 0。在 Checkpoint 5 中 `false_hits > 0` 是 **blocker**（不是 suggestion），不解决不推进 |
| **AOT 信息复用** | 利用 AOT 已有的 divisibility/contiguity/size-type 信息，不做重复推理，降低出错概率 |
| **宁缺毋滥** | 若符号信息不足以 100% 确定特化条件满足，就回退到通用路径。漏过 1 个可特化 case 只丢 ~0.83 分（1/12 × 10），错误特化 1 个 case 扣 1 分 **且** 总分还要跟扣分后的分数折算 |

#### 谓词设计原则

```
正确谓词: "对所有满足 P(input) 的输入，特化路径与通用路径语义等价"
错误谓词: "对常见的满足 P(input) 的输入，特化路径看起来是对的"

正确谓词: size % tile_size == 0  (符号可判定)
错误谓词: size == 1024           (硬编码)
错误谓词: test_name == "test_add" (识别测试名)
```

#### Loop 迭代策略

```
while not Checkpoint 3 passed:
    design = 生成谓词设计文档
    verify: 手动挑 5 个应命中的输入 + 5 个不应命中的输入
    verify: 谓词不含硬编码尺寸/文件名
    if 任何问题 → 重设计
    else → record_result(PASS) → advance to Phase 4
```

---

### 2.3 Generated Code Metric — 20/20

**评分规则**: 8 个隐藏 generated source 结构用例。每个用例计算 `reduction = (baseline_count - submitted_count) / baseline_count`。reduction ≥ 0.25 满分，0-0.25 线性计分，≤0 得 0 分。8 个用例取平均折算到 20 分。

#### Loop Orchestrator 的角色

```
Phase 6: Benchmark
  ├── BenchmarkRunner.run_all()
  ├── BenchmarkRunner.analyze_generated_source()
  └── Checkpoint 6: avg_reduction >= 0.25
```

#### 满分策略

| 策略 | 具体做法 | 目标 reduction |
|------|---------|---------------|
| **Divisible tile: mask 归零** | 整除时跳过所有 mask 生成。baseline 有 6 个 mask 子条件 → submitted 有 0 个 | mask_expr_count reduction = 1.0 |
| **Contiguous: stride 归零** | 连续布局时消除 stride 乘法。baseline 有 ndim 个 stride 项 → submitted 有 0 个 | stride_expr_count reduction = 1.0 |
| **组合收益** | 既整除又连续时，mask **且** stride 都归零，source_line_count 显著减少 | source_line_count reduction ≥ 0.30 |
| **选对特化类别** | Divisible tile + Contiguous 的组合天然产出高 reduction。Broadcast/scalar 的收益较分散 | — |

#### 量化预期

| 场景 | baseline mask | submitted mask | baseline stride | submitted stride | 综合 reduction |
|------|-------------|----------------|-----------------|-------------------|---------------|
| 整除 + 连续 (element-wise 1D, size=1024, tile=256) | 6 | 0 | 3 | 0 | ~0.40 |
| 整除 + 连续 (matmul M=N=K=256, tile=64) | 6 | 0 | 3 | 0 | ~0.35 |
| 整除 + 非连续 (sliced tensor, size=1024) | 6 | 0 | 3 | 3 | ~0.20 |
| 非整除 + 连续 (size=1000, tile=256) | 6 | 6 | 3 | 0 | ~0.15 |
| 非整除 + 非连续 (最坏 fallback) | 6 | 6 | 3 | 3 | 0.00 |

**关键洞察**: 只有当两个特化同时命中时 reduction 才远超 0.25。若只实现 1 个特化，reduction 可能在 0.15-0.25 之间震荡——不够稳。**建议实现两个特化**。

#### Loop 迭代策略

```
while not Checkpoint 6 passed:
    results = BenchmarkRunner.run_all(SPEC_HIT_CASES + FALLBACK_CASES)
    avg_reduction = mean(r.code_metrics.mask_expr_count.reduction for r in results if r.specialization_hit)
    
    if avg_reduction < 0.25:
        → 分析哪些 case 未达预期
        → 如果 mask 未归零: 检查 divisible tile 谓词是否过紧
        → 如果 stride 未归零: 检查 contiguous 谓词是否过紧
        → 如果两个特化都命中了但 reduction 仍低: 检查 source 分析器的计数逻辑
        → 回到 Phase 4 修复
    else:
        → record_result(PASS) → advance to Phase 7
```

---

### 2.4 Runtime — 20/20

**评分规则**: 8 个隐藏 benchmark 场景。speedup ≥ 1.10 满分，1.00-1.10 线性计分，0.95-1.00 得 30%，<0.95 得 0 分（并记录性能回退）。

#### Loop Orchestrator 的角色

```
Phase 6: Benchmark
  ├── BenchmarkRunner.run_all()
  └── Checkpoint 6: avg_speedup >= 1.10, zero regressions
```

#### 满分策略

| 策略 | 具体做法 | 预期收益 |
|------|---------|---------|
| **Mask 消除 = 减少分支** | 无 mask 的 `tl.load`/`tl.store` 比有 mask 的快。GPU warp 中分支 divergence 是性能杀手 | 5-15% |
| **Stride 消除 = 减少乘法** | 消除 `stride * offset` 乘法指令。对 memory-bound kernel 收益较小但对 compute-bound kernel 明显 | 2-8% |
| **组合收益** | 同时消除 mask 和 stride 时，指令数减少 + 分支减少 → 叠加收益 | 10-20% |
| **Fallback 零退化** | 条件不满足时走原路径，代码与 baseline 完全一致 → speedup = 1.00 | 不扣分 |
| **Warmup + 多次测量** | BenchmarkRunner 默认 warmup=3 + trials=10，取中位数而非均值以抗噪声 | 数据稳定 |

#### 收益分解

| 优化 | 对什么 kernel 最有效 | 预期 speedup |
|------|-------------------|-------------|
| Mask 消除 | Memory-bound kernel（element-wise、copy）| 1.05-1.12 |
| Stride 消除 | Compute-bound kernel（matmul、conv）| 1.02-1.08 |
| 两者叠加 | 任何整除 + 连续的 kernel | 1.10-1.20 |

#### 性能回退防护

```
Checkpoint 6 硬性规则:
  - 任何 case 的 speedup < 0.95 → BLOCKER（不是 suggestion）
  - 任何 case 的 speedup < 1.00 → 必须写分析报告
  - Fallback case 的 speedup 必须在 0.98-1.02 范围内（验证代码未退化）
```

#### Loop 迭代策略

```
while not Checkpoint 6 passed:
    results = BenchmarkRunner.run_all()
    
    for r in results:
        if r.speedup < 0.95:
            → 🔴 回到 Phase 4: 检查特化路径是否正确
            → 可能原因: 特化路径的 pointer arithmetic 方式变了导致 bank conflict
        if r.speedup < 1.00 and r.specialization_hit:
            → 🟡 特化路径反而更慢 — 检查是否引入了不必要的同步或寄存器压力
        if r.speedup < 0.98 and not r.specialization_hit:
            → 🟡 Fallback 性能退化 — 代码路径可能被意外修改
    
    avg = mean(r.speedup for r in results if r.specialization_hit)
    if avg >= 1.10:
        → record_result(PASS) → advance to Phase 7
    else:
        → 分析瓶颈: 哪些 case 拉了后腿？
        → 如果所有 hit case 的 speedup 都在 1.05-1.10 之间:
            → 考虑实现第二个特化以叠加收益
        → 如果个别 case 的 speedup < 1.00:
            → 回到 Phase 4 深度分析该 case 的 generated source
```

---

### 2.5 工程与报告质量 — 10/10

**评分标准**: weakness analysis、fallback 设计、代码边界、测试质量、无硬编码和引用披露。

#### Loop Orchestrator 的角色

```
Phase 7: Report
  ├── 生成赛题报告 PDF
  ├── 生成 HONOR_CODE.md
  ├── 生成 REFERENCE.md
  └── Checkpoint 7: 全部 4 项齐备
```

#### 满分策略

| 评分项 | 如何拿满分 | 存放位置 |
|--------|----------|---------|
| **Weakness analysis** | Phase 2 输出 `docs/weakness_analysis.md`。必须包含：具体场景、baseline 源码片段、量化指标、低效点归类、预期改进形态 | `aicamp2026/docs/weakness_analysis.md` |
| **Fallback 设计** | Phase 3 输出 `docs/specialization_design.md`。必须包含：布尔谓词、if-else 分支位置、不为特化时的行为、为什么 fallback 是安全的 | `aicamp2026/docs/specialization_design.md` |
| **代码边界** | 改动集中在 generation.py + tensor.py 的少数函数。diff 小而聚焦 → 评审加分 | `ninetoothed/src/ninetoothed/` |
| **测试质量** | 既有测试不改动 + 新增 ≥6 个针对性测试 + 对抗性场景。测试命名清晰、有 docstring | `ninetoothed/tests/` |
| **无硬编码** | 全量 diff review 确保无文件名/尺寸/benchmark 名硬编码 | Checkpoint 4 + 8 |
| **引用披露** | REFERENCE.md 列出所有参考的论文/文档/代码/工具。AI 辅助使用在 HONOR_CODE.md 中披露 | `aicamp2026/report/` |

---

## 三、必须交付物 × Loop Orchestrator 产出对照

### 3.1 代码改动 (`generation.py` / `aot.py` / `tensor.py`)

| 交付要求 | Loop 如何系统化达成 |
|---------|-------------------|
| 1-2 类特化 | Phase 3 选定 → Phase 4 实现。Orchestrator 在 Phase 3 的 prompt 中明确要求：选定类别 + 写谓词 + 画改动位置 |
| 启用条件明确 | Checkpoint 3 验证: `predicates_written == True` |
| Fallback 保证 | Checkpoint 4 验证: `fallback_verified == True`（手动构造反例确认走原路径） |
| 不弱化既有测试 | Checkpoint 4 验证: `pytest tests/` 全部通过。Orchestrator 在 Phase 4 prompt 中明确规则：只加 if-else、不删不改原逻辑 |

### 3.2 测试 (≥6 个新测试)

| 测试类型 | 最少数量 | Loop 如何系统化生成 |
|---------|---------|-------------------|
| Specialization hit 测试 | 2 | Phase 5: TestEval.check_specialization() 驱动。Orchestrator prompt 模板指导 AI 构造满足谓词的输入 |
| Fallback correctness 测试 | 2 | Phase 5: TestEval.check_fallback() 驱动。构造接近但不满足谓词的边界输入 |
| Generated source 结构测试 | 2 | Phase 5: TestEval.check_generated_source() 驱动。检查 mask/stride/pointer 计数 |
| 对抗性场景测试 | ≥5/特化 | Phase 5.0: 在 Checkpoint 5 中是 blocker（`adversarial_passed`）|

### 3.3 Benchmark 数据 (JSON/CSV)

```
Phase 6 产出:
  ├── results/benchmark_YYYYMMDD_HHMMSS.json   ← BenchmarkRunner.export_json()
  ├── results/benchmark_YYYYMMDD_HHMMSS.csv    ← BenchmarkRunner.export_csv()
  └── 每个 case 含:
      ├── baseline_runtime_ms
      ├── submitted_runtime_ms
      ├── speedup
      ├── specialization_hit (bool)
      ├── mask_expr_count: {baseline, submitted, reduction}
      ├── stride_expr_count: {baseline, submitted, reduction}
      ├── pointer_expr_count: {baseline, submitted, reduction}
      └── source_line_count: {baseline, submitted, reduction}
```

### 3.4 赛题报告 (PDF)

```
aicamp2026/report/
├── Simon-CHOU_九齿编译优化_T1-2-1_赛题报告.pdf   ← 6 节完整内容
├── HONOR_CODE.md                                ← 签名 + AI 披露
└── REFERENCE.md                                 ← 引用清单
```

**报告 6 节内容 × Loop 数据来源**:

| 报告章节 | Loop 数据来源 |
|---------|-------------|
| 1. 功能概述与改动范围 | Phase 2 weakness_analysis.md + Phase 3 specialization_design.md |
| 2. 技术方案、核心设计和关键代码路径 | Phase 3 设计文档 + Phase 4 实现的 diff |
| 3. 正确性验证方法与结果 | Phase 5 TestEval.evaluate() 的 report() 输出 |
| 4. 指标、测试矩阵和对比数据 | Phase 6 Benchmark 的 JSON 导出 + 对比表 |
| 5. 性能回退、失败用例和不支持场景 | Phase 6 的回归分析 + Phase 5 的已知失败 |
| 6. 参考资料、第三方工具和 AI 辅助 | Phase 7 的 HONOR_CODE.md + REFERENCE.md |

---

## 四、Loop Orchestrator 全流程执行剧本

### Phase 0: 基础设施搭建 ✅ DONE

```
Orchestrator 输出: Phase 0 prompt → 验证 benchmark/pitfall/loop 可 import
Checkpoint 0: ✅ PASSED（所有包可 import，pitfall 解析 14 条记录）
```

### Phase 1: 环境搭建 + 基线（预计 0.5 天）

```
Orchestrator 输出: Phase 1 prompt
  → AI 执行: pytest tests/ -x --timeout=300
  → AI 执行: 收集 2-3 个典型算子的 baseline generated source
  → AI 记录: mask/stride/pointer 基线计数到 results/baselines/
  → AI 记录: pitfall 中的任何新环境问题

Checkpoint 1 通过条件:
  - pytest 全部通过（允许 known WONTFIX: test_jagged.py, test_attention.py fp32, bf16）
  - Triton 可 import, torch.cuda.is_available() = True
  - Baselines 已记录到 results/baselines/

当前已有机器的已知问题（来自 pitfall log）:
  - test_jagged.py: WONTFIX (nested tensor API)
  - test_attention.py fp32: WONTFIX (Triton 3.1.0 + SM 7.5 bug)
  - bf16: WONTFIX (RTX 2060 = SM 7.5 硬件限制)
```

### Phase 2: Weakness Analysis（预计 1-1.5 天）

```
Orchestrator 输出: Phase 2 prompt
  → AI 深入阅读以下关键函数:
    - generation.py::_generate_pointers_and_mask()
    - generation.py::_generate_overall_offsets_and_mask()
    - generation.py::_generate_offsets_and_mask()
    - generation.py::_generate_innermost_indices()
    - tensor.py::Tensor.offsets()  (L571-584 的 mask 累积逻辑)
    - aot.py::_enumerate_variant_specs()
    - aot.py::_build_variant()

  → AI 做快速原型验证:
    - 验证 is_contiguous 在符号 Tensor 上是否可实现
      (检查 stride[i] == prod(shape[i+1:]) 的 sympy 符号等价性)
    - 验证 mask 生成的准确入侵点
    - 验证 _generate_overall_offsets_and_mask 在 contiguous 场景的行为

  → AI 构造 ≥2 个弱势场景（从以下 4 个中至少选 2 个）:
    1. element-wise (flatten + tile): stride 已通过 flatten 消除，
       但 _generate_overall_offsets_and_mask 仍遍历 source.ndim
       个维度生成 stride 表达式 → stride_expr_count 目标降为 0
    2. matmul with M/N/K 整除 block_size: 尾块 mask 始终生成，
       即使整除时永远为 True → mask_expr_count 目标降为 0
    3. add 的 alpha 标量参数: 标量不需要 pointer/stride/mask，
       但当前仍生成完整路径 → pointer_expr_count 目标降为 0
    4. conv2d 的 constexpr stride/padding: 编译期已知参数未用于
       简化 mask/pointer → 待分析

  → AI 输出: docs/weakness_analysis.md
    每个 case 必须包含:
    - 具体场景描述（输入 shape、dtype、tile size）
    - Baseline 生成的源码片段（低效部分高亮标注）
    - 基线量化指标（mask_expr_count / stride_expr_count / pointer_expr_count）
    - 理论最优指标（优化后的预期值）
    - 低效点归类（冗余 mask / 冗余 stride / 冗余指针 / 未命中 variant / 广播未简化）
    - 预期的改进后源码形态（伪代码或简化后的 Triton 源码）

Checkpoint 2 通过条件:
  - ≥2 weakness cases 完成分析，每个有完整的量化基线 + 理论最优
  - is_contiguous 原型验证有明确结论（可行/不可行/降级方案）
  - docs/weakness_analysis.md 已输出
  - 本阶段发现的任何新问题已记录到 pitfall log
```

### Phase 3: 设计选择（预计 0.5-1 天）

```
Orchestrator 输出: Phase 3 prompt

  → AI 基于 Phase 2 的 weakness analysis 选择 1-2 个特化类别:

  **强烈推荐组合: Divisible tile (主) + Contiguous (副)**

  理由:
    - Divisible tile: 改动点在 tensor.py 的 mask 累积逻辑，风险最低，
      收益可直接量化（mask 表达式数量降为 0），AOT 已有 divisibility 信息可复用
    - Contiguous: 改动点在 generation.py 的 stride 处理，与 Tensor 的
      flatten 操作天然契合
    - 两个同时命中时: mask + stride 双双归零，reduction 远超 0.25 满分线
    - 特化交互清晰: Divisible tile 优先（减 mask）→ Contiguous（减 stride），
      两者独立作用在不同代码生成阶段，不冲突
    - 实现顺序: 先 Divisible tile → 完成测试和 benchmark 闭环 → 再 Contiguous

  → AI 为每个特化写布尔谓词:

    特化 1 (Divisible tile fast path):
      谓词: 对每一维 i，tile_size[i] 整除 total_size[i]
      形式: all(size[i] % tile_size[i] == 0 for i in range(ndim))
      判定: 基于 AOT divisibility 信息 + symbol 符号推理
      入侵点: Tensor.offsets() mask 累积逻辑 (tensor.py L571-584)

    特化 2 (Contiguous fast path):
      谓词: is_contiguous(tensor)
      形式: all(stride[i] == prod(shape[i+1:]) for i in range(ndim-1))
      判定: sympy 符号等价性推理（Phase 2.3 已验证可行性）
      入侵点: _generate_overall_offsets_and_mask() stride 处理

  → AI 设计特化交互规则:
    - 两个特化独立作用（mask 阶段 vs stride 阶段），可同时命中，不冲突
    - 任一不满足 → 该特化跳过，另一仍可命中
    - 优先级: Divisible tile > Contiguous（但它们独立所以其实无竞争）

  → AI 设计 fallback 保证:
    - generation.py 中保留通用路径不变
    - 特化路径作为 if-else 分支挂载
    - else 分支指向原通用路径（代码一字不改）
    - debug 模式下添加 assertion 验证特化路径与通用路径结果一致

  → AI 输出: docs/specialization_design.md
    必须包含:
    - 选定的特化类别和选择理由
    - 每个特化的布尔谓词（精确到可写进代码的形式）
    - 改动位置（文件 + 函数 + 行号范围）
    - if-else 分支结构的伪代码
    - 特化交互矩阵（两个特化同时命中的行为）
    - Fallback 路径描述（为什么安全）
    - 对照评分标准的速度/减量预期

Checkpoint 3 通过条件:
  - 1-2 个特化类别已选定，选择理由明确
  - 布尔谓词已写定（精确形式）
  - Fallback 路径已文档化
  - 设计不含硬编码（无固定尺寸、文件名、benchmark 名）
  - 评分目标确认: speedup ≥ 1.10, reduction ≥ 0.25
```

### Phase 4: 实现（预计 2-4 天）

```
Orchestrator 输出: Phase 4 prompt
  → AI 按 Phase 3 设计文档逐特化实现:

  **先实现特化 1 (Divisible tile)**:
    1. 在 Tensor.offsets() 中添加 divisibility 检查分支
    2. 若整除: 跳过所有 mask 子条件生成，直接设置 mask = True
    3. 在 _generate_offsets_and_mask() 中接收 divisibility 信息
    4. 复用 AOT 已有的 divisibility 信息（来自 _enumerate_variant_specs）
    5. if-else 清晰: if divisible → fast path else → 原路径（不变）

  **完成特化 1 后**:
    - rm -rf ~/.ninetoothed/  (清除编译缓存！)
    - pytest tests/ -x --timeout=300
    - 手动验证: 构造 3 个应命中 + 3 个应回退的输入
    - 若全部通过 → 记录 pitfall（如有新问题） → 继续特化 2

  **再实现特化 2 (Contiguous)**:
    1. 在 Tensor 类中添加 is_contiguous 属性（符号推理版）
    2. 在 _generate_overall_offsets_and_mask() 中检查 contiguity
    3. 若连续: stride = 1（跳过 stride 乘法）
    4. else: 走原路径（不变）

  **实现守则**（Orchestrator Phase 4 prompt 中的硬规则）:
    - ⛔ 不删除、跳过、弱化任何既有测试
    - ⛔ 不修改现有测试的断言逻辑
    - ✅ 每改一处 → 跑 pytest → 确认通过 → 再改下一处
    - ✅ 特化路径作为 if-else 分支挂载，else 永远指向原代码
    - ✅ 无硬编码尺寸/文件名/benchmark 名

  **实现自查 (Phase 4.4, 0.5 天)**:
    - [ ] 完整 diff review: 逐行检查所有修改
    - [ ] ruff format --check . && ruff check . 通过
    - [ ] pytest tests/ -x --timeout=300 全部通过
    - [ ] 所有 if-else 的 else 指向原通用路径（验证未改动原代码）
    - [ ] 无硬编码
    - [ ] 特化条件在 3 hit + 3 fallback 输入上行为正确

Checkpoint 4 通过条件:
  - pytest tests/ 全部通过（允许 known WONTFIX）
  - Fallback 路径手动验证: 非特化输入生成的代码与 baseline 一致
  - 特化命中验证: 特化输入确实触发快速路径
  - ruff 检查通过
  - 无硬编码
  - 编译缓存已清除
```

### Phase 5: 测试（预计 1-2 天）

```
Orchestrator 输出: Phase 5 prompt

  → AI 优先设计对抗性场景 (Phase 5.0, 0.5-1 天):
    **正确性门槛 29/30 意味着最多只能错 1 个隐藏用例**

    每特化至少 5 个对抗性 case:
      1. 整除边界: size=17 tile=16（刚好差 1）、size=16 tile=16（刚好整除）
      2. 非整除常规: size=1000 tile=256
      3. 多维混合: (128, 64) with tile (64, 32)，部分整除部分不整除
      4. 空/单维度: size=1、ndim=1
      5. dtype 混合: float32 + float16 + int32
      6. 极端尺寸: size=65536（大）、size=1（小）

    防误命中验证:
      - size=255 tile=256: 接近但不整除 → 不应命中 divisible tile
      - stride 近似连续但非连续 → 不应命中 contiguous
      - 每个接近但不满足的输入验证确实走了 fallback 路径
      - **false_hits == 0 是 Phase 5 的 BLOCKER**

  → AI 编写新测试 (≥6 个):

    Specialization hit 测试 (≥2):
      1. test_divisible_tile_hit_add: size=1024 tile=256 float32 add,
         验证生成的 Triton 源码中 mask 表达式数量 = 0
      2. test_contiguous_hit_add: contiguous 2D tensor add,
         验证生成的 Triton 源码中 stride 表达式数量 = 0

    Fallback correctness 测试 (≥2):
      1. test_non_divisible_fallback_add: size=1000 tile=256,
         验证计算结果与 torch.add 一致
      2. test_non_contiguous_fallback_add: sliced tensor (非连续),
         验证计算结果与 torch.add 一致

    Generated source 结构测试 (≥2):
      1. test_source_mask_expr_count: 整除/非整除时 mask 数量是否符合预期
      2. test_source_stride_expr_count: 连续/非连续时 stride 表达式数量

  → 运行 ntops 全量测试:
    cd <workspace>/ntops && pytest tests/
    确保 ninetoothed 改动不破坏任何上游算子

Checkpoint 5 通过条件:
  - ≥2 hit + ≥2 fallback + ≥2 source structure = ≥6 个新测试全部通过
  - 既有测试全部通过（无退化）
  - 对抗性场景: ≥5 个/特化全部通过
  - **false_hits == 0**（这是 blocker，不解决不推进）
  - ntops 全量测试通过
  - 本阶段发现的 bug 已记录到 pitfall log
```

### Phase 6: Benchmark（预计 0.5-1 天）

```
Orchestrator 输出: Phase 6 prompt
  → AI 使用 BenchmarkRunner 收集全量数据:

  SPEC_HIT_CASES:
    1. DIVISIBLE_TILE_ADD_1024:    size=1024 tile=256, float32, 1D add
    2. DIVISIBLE_TILE_MUL_4096:    size=4096 tile=256, float32, 1D mul
    3. CONTIGUOUS_ADD_2D:          shape=(256,128), contiguous, float32 add
    4. CONTIGUOUS_ADD_1D_F16:      size=2048, contiguous, float16 add

  FALLBACK_CASES:
    1. NON_DIVISIBLE_ADD_1000:     size=1000 tile=256, float32 add
    2. NON_DIVISIBLE_MUL_5000:     size=5000 tile=256, float32 mul
    3. NON_CONTIGUOUS_SLICED:      sliced tensor, float32 add
    4. NON_CONTIGUOUS_TRANSPOSED:  transposed 2D, float32 add

  → BenchmarkRunner 自动:
    1. 编译每个 case 的 kernel（通过 ninetoothed.jit）
    2. 捕获 generated Triton source
    3. analyze_generated_source(): 统计 mask/stride/pointer 计数
    4. Warmup 3 次 + 测量 10 次
    5. 计算 speedup + reduction
    6. export_json() + export_csv()

  → AI 分析结果:
    - 若任何 speedup < 0.95 → 🔴 BLOCKER → 回到 Phase 4
    - 若 avg_reduction < 0.25 → 🟡 → 分析哪些 case 拉低平均值
    - 若 avg_speedup < 1.10 → 🟡 → 分析是否需要第二个特化叠加

  → 期望结果:
    - 两个特化都命中时: reduction ≈ 0.35-0.40 (>> 0.25)
    - 两个特化都命中时: speedup ≈ 1.10-1.20 (>= 1.10)
    - Fallback case: speedup ≈ 1.00 (无退化)

Checkpoint 6 通过条件:
  - ≥2 hit + ≥2 fallback case 的完整 benchmark 数据
  - **speedup < 0.95 的 case 数量 = 0**（这是 blocker）
  - avg reduction ≥ 0.25（满分标准）
  - avg speedup ≥ 1.10（满分标准）
  - Benchmark JSON/CSV 已输出到 results/
```

### Phase 7: 报告与合规（预计 1-2 天）

```
Orchestrator 输出: Phase 7 prompt
  → AI 在 aicamp2026/report/ 下准备全部交付物:

  **1. 赛题报告 PDF**
     文件名: Simon-CHOU_九齿编译优化_T1-2-1_赛题报告.pdf

     第 1 节 - 功能概述与改动范围:
       来源: Phase 2 weakness_analysis.md + Phase 3 specialization_design.md
       内容: 赛题背景、选择 Divisible tile + Contiguous 的理由、
             改动范围（generation.py / tensor.py 的具体函数）

     第 2 节 - 技术方案、核心设计和关键代码路径:
       来源: Phase 3 specialization_design.md + Phase 4 实现 diff
       内容: 布尔谓词定义、if-else 分支结构、核心代码路径描述、
             **特化的正式正确性论证**（为什么特化是语义等价的）

     第 3 节 - 正确性验证方法与结果:
       来源: Phase 5 TestEval.evaluate() 的 report() 输出
       内容: pytest 结果、specialization hit/fallback 验证、
             对抗性场景测试结果、防误命中验证结果

     第 4 节 - 指标、测试矩阵和对比数据:
       来源: Phase 6 Benchmark JSON 导出
       内容: 指标对比表（baseline vs submitted）、speedup 分布、
             reduction 分布、测试矩阵

     第 5 节 - 性能回退、失败用例和不支持场景:
       来源: Phase 6 回归分析 + pitfall log
       内容: 任何 speedup < 1.0 的分析、已知 WONTFIX（test_jagged、
             test_attention fp32、bf16）、不支持场景说明

     第 6 节 - 参考资料、第三方工具和 AI 辅助:
       来源: REFERENCE.md + HONOR_CODE.md
       内容: 参考的论文/文档/代码、使用的工具（pytest/ruff/triton）、
             AI 辅助使用方式披露

  **2. HONOR_CODE.md**
     - 独立完成范围声明（哪些是自己写的，哪些基于已有代码）
     - AI 辅助使用情况: Claude Code + loop orchestrator
       （工具、使用范围、使用方式）
     - 外部代码引用列表
     - 签名 + 日期

  **3. REFERENCE.md**
     - 参考资料: NineToothed 源码和文档、Triton 语言规范、
       sympy 文档、CUDA 编程指南
     - 参考实现: ninetoothed 主库 (github.com/InfiniTensor/ninetoothed)
     - 外部工具: pytest, ruff, triton, torch, sympy

  **4. PR 描述（7 项）**
     1. 赛题编号与小组名称: T1-2-1 / Simon-CHOU
     2. 主要改动点: generation.py (contiguous fast path) +
        tensor.py (divisible tile mask logic)
     3. 自测命令和结果: pytest tests/ 输出截图 + 环境信息
     4. 指标对比表: 从 Benchmark JSON 生成
     5. 未覆盖/已知风险: test_jagged WONTFIX 等
     6. HONOR_CODE.md + REFERENCE.md 链接
     7. 赛题报告链接

Checkpoint 7 通过条件:
  - 赛题报告 PDF 6 项内容齐全
  - HONOR_CODE.md 已署名，AI 使用已披露
  - REFERENCE.md 引用完整
  - PR 描述 7 项内容齐全
  - 所有交叉引用链接有效
```

### Phase 8: 提交（预计 0.5 天）

```
Orchestrator 输出: Phase 8 prompt
  → AI 执行最终检查:

    最终 checklist:
    - [ ] pytest tests/ 全部通过（无跳过、无弱化）
    - [ ] ruff format --check . && ruff check . 通过
    - [ ] 无硬编码尺寸/文件名/benchmark 名
    - [ ] 无未声明的外部依赖
    - [ ] 所有新增代码有对应的测试覆盖
    - [ ] HONOR_CODE.md + REFERENCE.md + 赛题报告 PDF 已包含
    - [ ] PR 描述包含全部 7 项内容
    - [ ] 分支名: 2026-spring-Simon-CHOU-T1-2-1 ✅
    - [ ] PR 标题: [2026春季][T1-2-1] Simon-CHOU
    - [ ] PR 目标: InfiniTensor/ninetoothed main
    - [ ] ntops 验证通过（无需单独 PR）

  → 创建 PR
  → 预留 7/12 全天处理 reviewer 反馈

Checkpoint 8 通过条件: 全部 checklist 通过
```

---

## 五、满分的关键杠杆点

### 满分公式

```
100 = 30(Correctness) + 20(Specialization) + 20(Code Metric) + 20(Runtime) + 10(Engineering)

Correctness 30:   对抗性测试全覆盖 + 零误命中 + ntops 兼容
Specialization 20: 精确谓词 + AOT 信息复用 + 宁缺毋滥
Code Metric 20:   两个特化叠加 (divisible + contiguous) → reduction ≥ 0.25
Runtime 20:       两个特化叠加 → speedup ≥ 1.10
Engineering 10:   weakness_analysis.md + specialization_design.md + 
                  HONOR_CODE.md + REFERENCE.md + 报告 PDF
```

### 最大风险点（按破坏性排序）

| 风险 | 破坏性 | 后果 | 缓解措施 |
|------|--------|------|---------|
| **Correctness < 29/30** | 🔴 致命 | 总分上限 40，满分直接不可能 | Phase 5 对抗性测试做到极致；防误命中验证全覆盖 |
| **特化误命中** | 🔴 严重 | 每次扣 1 分 + 覆盖率折算 | 谓词收紧 + 符号验证；`false_hits == 0` 是 blocker |
| **只做一个特化** | 🟡 中等 | reduction 可能不达 0.25，speedup 可能不达 1.10 | 坚持 Divisible tile + Contiguous 双特化 |
| **编译缓存未清除** | 🟡 中等 | 测试的是旧代码，bug 被掩盖 | Phase 4 每次改动后 `rm -rf ~/.ninetoothed/` |
| **隐藏测试翻车** | 🟡 中等 | 边缘 case 没覆盖到 | 对抗性场景设计覆盖除法边界、多维混合、极端尺寸 |
| **报告缺项** | 🟢 轻度 | 工程质量扣分 | Phase 7 对照 checklist 逐项检查 |

---

## 六、Loop Orchestrator 的 AI 协作模式

### 每次 Phase 切换时的标准流程

```
1. orchestrator.checkpoint() → (passed, blockers, suggestions)
2. if not passed:
     → 留在当前 Phase
     → orchestrator.generate_next_action() → 新的 AI prompt
     → AI 执行 prompt → orchestrator.record_result()
     → 回到步骤 1
3. if passed:
     → orchestrator.advance_phase()
     → 新 Phase 的 orchestrator.generate_next_action()
     → AI 执行新的 prompt
```

### AI 在每个 Phase 拿到什么上下文

通过 `orchestrator.get_context_for_ai()`，AI 获得：
- **当前 Phase 名称和编号**: 知道自己该做什么
- **Tasking 上下文**: 从 tasking.md 提取的当前 Phase 章节
- **Testeval 结果**: 最近的 pytest/specialization/fallback 测试数据
- **Benchmark 结果**: 最近的 speedup/reduction 数据
- **Pitfall 统计**: 未解决的问题列表（避免重复踩坑）
- **Exit criteria**: 当前的 blockers 和 suggestions
- **Prompt 模板**: 该 Phase 的具体任务指令

### 人工介入点

Loop orchestrator 不是全自动的——以下节点需要人工决策：
1. Phase 3 设计选择: AI 推荐特化类别，人工确认
2. Phase 4 实现: AI 生成代码，人工 review diff
3. Phase 5 测试: AI 跑测试，人工确认对抗性 case 的充分性
4. Phase 6 Benchmark: AI 收集数据，人工确认 speedup/reduction 是否满意
5. Phase 7 报告: AI 生成草稿，人工审阅和补充
6. Phase 8 提交: 人工点击 PR 创建按钮

---

## 七、一句话总结

**Loop orchestrator 的本质是把赛题评分标准工程化为可执行的 checkpoint 序列。每通过一个 checkpoint，就离满分近一步。checkpoint 不通过就不推进——这个机制保证了不会带着未解决的问题进入下一阶段，从而系统性降低"隐藏测试翻车"的风险。**
