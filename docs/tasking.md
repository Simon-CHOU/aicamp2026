# T1-2-1 任务拆解与执行计划

**赛题**: 九齿编译优化 T1-2-1 — NineToothed 代码生成特化增强挑战  
**GitHub ID**: Simon-CHOU  
**日期**: 2026-06-25  
**最终截止**: 2026-07-13 0:00（即 7/12 午夜，剩余约 18 天）  
**正确性门槛**: 隐藏 correctness **29/30**（未达标总分上限 40/100）  
**隐藏评测规模**: 30 correctness + 12 specialization + 8 code metric + 8 benchmark = **58 个用例**

---

## 当前状态速览

| 项目 | 路径 | 角色 |
|------|------|------|
| ninetoothed (主库 fork) | `D:\ml\ninetooth2026\ninetoothed` | 修改目标——赛题代码改动在此 |
| ntops (算子库 fork) | `D:\ml\ninetooth2026\ntops` | 验证试验场——用其测试和 benchmark |

**已有基础设施**（ninetoothed 主库已包含）:
- AOT 层面已有 divisibility / contiguity / size-type 特化 (`aot.py`)
- `generation.py` 的 `CodeGenerator` 已处理 pointer/mask/offset 生成
- `Tensor` 类已有元信息查询（shape, stride, ndim, jagged_dim 等）
- CI 已有 pytest 在 NVIDIA self-hosted runner 上运行

**关键发现**: 现有特化主要在 AOT dispatch 层面（选择哪个 variant），而 Triton 源码生成 (`generation.py`) 总是走最通用的路径——这正是赛题要求改进的核心。

---

## Phase 1: 环境搭建与基线确认（预计 1-2 天）

> **Exit Criteria**: `pytest tests/` 全部通过；CUDA 可用；Triton 编译器可正常生成 kernel；baseline 指标已记录

### 1.1 创建开发分支
- 分支命名: `2026-spring-Simon-CHOU-T1-2-1`
- 基于当前 master (`ef4c528`) 创建

### 1.2 安装开发环境
- **环境预检**: 
  - 确认 `torch.cuda.is_available()` 为 True
  - 确认 `triton` 可 import 且可编译 kernel
  - **若 Windows 原生不支持 Triton**: 立即切换到 WSL2 或 Linux 双系统/远程 GPU 实例
  - 冻结 triton 版本（固定版本号或 commit hash），确保与 CI 一致
- 在 ninetoothed 目录 `pip install -e .`（可编辑安装）
- 在 ntops 目录 `pip install -e .`
- 确认 `pytest` 可以运行 CUDA 测试
- 运行 `ruff format --check . && ruff check .` 确认代码风格基线

### 1.3 运行全量基线测试
- `pytest tests/ -x --timeout=300` 记录全部通过
- 记录测试数量、跳过数量、失败数量作为基线
- 特别关注 `test_generation.py` 和 `test_aot.py`

### 1.4 收集基线 generated source
- 选取 2-3 个典型算子（如 element-wise add、matmul），捕获其生成的 Triton 源码
- 记录 mask 数量、stride 表达式数量、pointer arithmetic 数量作为对比基线

---

## Phase 2: Weakness Analysis（预计 2-3 天）

> **Exit Criteria**: 
> - ≥2 个 weakness case 完成分析，含 baseline 源码片段和量化指标（mask_expr_count / stride_expr_count / pointer_expr_count）
> - 每个 case 的低效点已归类（冗余 mask / 冗余 stride / 冗余 pointer arithmetic / 未命中特化 variant / 广播未简化）
> - 每个 case 有预期的优化后源码形态和理论最优指标
> - `docs/weakness_analysis.md` 已输出
> - `is_contiguous` 在符号 Tensor 上的可行性已通过快速原型验证

### 2.1 深入理解 generation.py 的代码生成路径
关键函数：
- `_generate_pointers_and_mask()` — 生成 pointer + mask
- `_generate_overall_offsets_and_mask()` — 生成 `sum(stride[dim] * offset[dim])` 形式的整体偏移
- `_generate_offsets_and_mask()` — 递归计算各层级的 offset 和 mask
- `_generate_innermost_indices()` — 生成最内层循环索引（arange）

重点关注：
- 何时 mask 是冗余的（tile 整除 total size 时）
- 何时 stride 乘法是冗余的（contiguous 布局时，每维 stride 就是 1 或可推导）
- 何时 pointer arithmetic 可简化（broadcast/scalar 场景）

### 2.2 深入理解 aot.py 的 variant 枚举
关键函数：
- `_enumerate_variant_specs()` — 枚举所有 variant 组合
- `_build_variant()` — 为每个 variant 构建编译产物
- `_generate_dispatcher()` — 生成 C++ dispatcher

已有特化：divisibility (size % 16 == 0)、contiguity (stride == 1)、int32/int64 size/stride

### 2.3 快速原型验证（新增）
在开始完整的 weakness analysis 之前，先做小范围代码实验确认关键假设：
- 验证 `is_contiguous` 在符号 Tensor 上的可行性：检查 `stride[i] == prod(shape[i+1:])` 的符号等价性推理是否可实现
- 验证 mask 生成的准确入侵点：trace `Tensor.offsets()` (tensor.py L571-584) 中 mask 累积逻辑，确认 divisible tile 特化的修改位置
- 验证 `_generate_overall_offsets_and_mask` 在 contiguous 场景下的行为

### 2.4 构造弱势场景并文档化（>= 2 个 case）

至少分析以下场景，找出 2+ 个明确弱势：

| 场景 | 潜在弱势 | 量化目标 |
|------|---------|---------|
| element-wise (flatten + tile) | stride 已通过 flatten 消除，但生成的代码仍包含 stride 计算 | stride_expr_count → 0 |
| matmul with M/N/K 整除 block_size | 尾块 mask 始终生成，即使整除时永远为 True | mask_expr_count → 0 |
| add 的 alpha 标量参数 | 标量不需要 pointer/stride/mask，但当前仍生成完整路径 | pointer_expr_count → 0 |
| conv2d 的 constexpr stride/padding | 编译期已知参数未用于简化 mask/pointer | 待分析 |

**注意**: element-wise flatten 场景中，flatten 改变了 offset 计算方式，但 `_generate_overall_offsets_and_mask` 仍遍历 `source.ndim` 个维度生成 stride 表达式。flatten 并未消除 stride——需要在 contiguous fast path 中专门处理。

### 2.5 编写 weakness analysis 文档
输出到 `docs/weakness_analysis.md`，包含：
- 每个 case 的具体场景描述
- baseline 生成的源码片段（低效部分高亮）
- **基线量化指标**（mask_expr_count / stride_expr_count / pointer_expr_count）
- **理论最优指标**（优化后的预期值）
- 低效点归类（冗余 mask / 冗余 stride / 冗余 pointer arithmetic / 未命中特化 variant / 广播未简化）
- 预期的改进后源码形态

---

## Phase 3: 选择特化类别与设计（预计 1-2 天）

> **Exit Criteria**: 
> - 已选定 1-2 个特化类别，每个有明确的设计文档
> - 每个特化的启用条件已用布尔谓词形式写出
> - 特化间的交互/优先级已明确（若选两个）
> - fallback 路径设计已文档化
> - 已对照赛题评分标准确认设计目标：speedup ≥ 1.10（满分）、code metric reduction ≥ 0.25（满分）
> - 设计不含硬编码尺寸、文件名、benchmark 名称

### 3.1 选择 1-2 个特化类别
基于 weakness analysis 结果，从以下 4 类中选择：

1. **Contiguous fast path** — 当 stride 信息可判定为连续访问时，简化 pointer 表达式
2. **Divisible tile fast path** — 当 tile 覆盖无尾块时，省略边界 mask
3. **Broadcast / scalar fast path** — 广播维度/size-1/标量简化
4. **Layout-known AOT variant** — 利用 AOT 已知信息选择更专门的 variant

**建议优先考虑**:
- **Divisible tile fast path**（Category 2）：改动点在 `Tensor.offsets()` 的 mask 累积逻辑（tensor.py L571-584），收益可直接量化（mask 表达式数量降为 0），已有 AOT divisibility 信息可复用
- **Contiguous fast path**（Category 1）：改动点在 `_generate_overall_offsets_and_mask` 的 stride 乘法消除，与 Tensor 的 flatten 操作天然契合

**实现顺序**: 先实现 1 个特化（建议 Divisible tile），完成测试和 benchmark 闭环后再扩展到第 2 个。不并行实现两个特化——降低调试复杂度。

**特化交互**: 若两个特化同时命中（如既 divisible 又 contiguous），定义优先级规则：Divisible tile 优先（减少 mask）→ Contiguous（减少 stride），两者独立作用在不同代码生成阶段，不冲突。

### 3.2 设计启用条件
每个特化必须：
- 有明确的、可验证的**布尔谓词**形式的启用条件
- 条件不满足时自动回退到通用路径
- 不能依赖测试文件名、benchmark 名称或固定尺寸
- 仅在符号信息足以**可靠判定**时启用（宁可不优化也不错优化）

### 3.3 设计 fallback 保证
- 在 `generation.py` 中保留通用路径不变
- 特化路径作为 if-else 分支挂载
- 确保不符合条件的输入走原路径
- 添加运行时 assertion（debug 模式）验证特化路径结果与通用路径一致

---

## Phase 4: 实现（预计 3-5 天）

> **Exit Criteria**: 
> - 特化代码已实现，if-else 分支清晰
> - `pytest tests/` 全部通过（无跳过、无弱化）
> - fallback 路径手动验证通过（生成的代码与 baseline 一致）
> - 特化 1 实现完成且自我测试通过后，才启动特化 2

### 4.1 实现特化逻辑

**特化 1（Divisible tile fast path）**:
- 修改 `Tensor.offsets()` (tensor.py) 的 mask 累积逻辑：当 tile 大小整除 total size 时，跳过 mask 生成（mask = True）
- 修改 `_generate_offsets_and_mask()` (generation.py)：接收 divisibility 信息，在生成 mask 前插入检查分支
- 复用 AOT 已有的 divisibility 信息

**特化 2（Contiguous fast path，如选择）**:
- 修改 `_generate_overall_offsets_and_mask()`：当判定为 contiguous 时，简化 stride 乘法（stride = 1 时消除乘法项）
- 在 `Tensor` 类中增加 helper 属性用于符号层面的 contiguity 判定
- **注意**: Tensor 是符号张量（shape/stride 含 sympy 表达式），`is_contiguous` 需做符号推理（检查 `stride[i] == prod(shape[i+1:])` 的符号等价性），不能直接套用 PyTorch 的实现。此逻辑需在 Phase 2.3 已验证可行性。

在 `src/ninetoothed/aot.py` 中（如选择 Category 4）：
- 增强 variant 枚举逻辑
- 传递更多元信息给 code generator

### 4.2 保持现有测试不变
- 不删除、跳过、弱化任何现有测试
- 不修改现有测试的断言逻辑
- 修改代码后反复运行 `pytest tests/` 确认全部通过
- **在实现过程中持续运行测试**，而非等到 Phase 5 才统一跑

### 4.3 验证 fallback 路径
- 手动构造不满足特化条件的输入
- 确认生成的代码与 baseline 一致（未退化）
- 确认功能正确性不受影响

### 4.4 实现自查（新增，0.5 天）
在进入 Phase 5 之前：
- [ ] 完整 diff review：逐行检查所有修改
- [ ] `ruff format --check . && ruff check .` 通过
- [ ] `pytest tests/ -x --timeout=300` 全部通过
- [ ] 确认所有 if-else 分支中 else 指向原通用路径
- [ ] 确认无硬编码尺寸、文件名、benchmark 名称
- [ ] 特化条件用布尔谓词验证：手动挑 3 个应命中的输入和 3 个应回退的输入确认行为

---

## Phase 5: 测试（预计 2-3 天）

> **Exit Criteria**: 
> - 所有新增测试通过（≥2 hit + ≥2 fallback + ≥2 structure = ≥6 个新测试）
> - 所有既有测试仍通过
> - ntops 全量测试通过
> - 对抗性场景测试通过（≥5 个边缘 case/特化）
> - 若发现 bug → 记录、修复、重新跑全量测试 → 回到 Phase 4.4 自查

### 5.0 对抗性场景设计（新增，0.5-1 天）
正确性门槛 29/30 意味着最多只能错 1 个隐藏用例。必须主动设计边缘 case：

**每个特化至少构造 5 个对抗性 case**：
- 整除/非整除边界（如 size=17，tile=16）
- 1D/2D/3D 混合维度
- 空维度 / size=1 维度
- 不同 dtype 组合（float32/float16/int32）
- 极端尺寸（非常大或非常小的 tensor）

**防误命中验证**（应对 specialization coverage 扣分规则）：
- 构造接近但不满足特化条件的输入，验证**不会**错误命中特化
- 每个应回退的用例错误命中特化在隐藏评测中扣 1 分

### 5.1 新增 specialization hit 测试（>= 2 个）
- 构造满足特化条件的输入
- 验证生成的 Triton 源码确实走了特化路径
- 例如：检查 mask 表达式数量是否为 0（divisible tile 特化）

### 5.2 新增 fallback correctness 测试（>= 2 个）
- 构造不满足特化条件的输入
- 验证计算结果与 PyTorch 参考实现一致
- 确保 fallback 路径仍然正确
- 包含接近但不满足特化条件的边界输入

### 5.3 新增 generated source 结构测试（>= 2 个）
检查项：
- `tl.load` / `tl.store` 的 mask 参数是否存在/不存在
- stride 表达式数量
- pointer arithmetic 复杂度
- variant 名称是否命中

### 5.4 运行 ntops 全量测试
- `cd D:\ml\ninetooth2026\ntops && pytest tests/`
- 确保 ninetoothed 的修改不破坏上游算子库
- **建议在 Phase 4 实现过程中就间歇性运行**，而非等到 Phase 5 最后

### 5.5 Bug 修复与迭代（新增）
本阶段发现的任何问题按以下流程处理：
1. 记录 bug 和复现步骤
2. 评估修复时间是否在缓冲范围内
3. 回到 Phase 4 修复 → 运行 Phase 4.4 自查 → 重新运行 Phase 5 测试
4. 若修复时间超出缓冲，降级为"已知风险"并在报告中说明

---

## Phase 6: Benchmark（预计 1-2 天）

> **Exit Criteria**: 
> - ≥2 hit + ≥2 fallback case 的 benchmark 数据已收集
> - 每个 case 有完整的指标记录
> - speedup ≥ 1.10 的 case 数量已知（赛题满分标准）
> - reduction ≥ 0.25 的 case 数量已知（赛题满分标准）
> - 性能回退 case（speedup < 0.95）已标记
> - 若发现回退 → 分析原因 → 决定是否回到 Phase 4 修复

### 6.1 编写 benchmark 脚本
- 至少 2 个命中特化的 case
- 至少 2 个不应命中特化的 fallback case
- 输出 JSON 或 CSV 格式

### 6.2 记录指标
每个场景至少记录：
- `baseline_runtime_ms`
- `submitted_runtime_ms`
- `speedup`（目标 ≥ 1.10 满分；1.00-1.10 线性计分；0.95-1.00 得 30%；< 0.95 得 0 分）
- `specialization_hit` (bool)
- 至少一个 generated code 指标：`mask_expr_count` / `stride_expr_count` / `pointer_expr_count` / `variant_name` / `source_line_count`（目标 reduction ≥ 0.25 满分；0-0.25 线性计分；≤ 0 得 0 分）

### 6.3 对比分析
- 生成指标对比表
- 标识性能回退的 case（speedup < 0.95）
- 分析未覆盖或未改善的场景
- 若发现 speedup < 1.0 或 reduction ≤ 0：分析根因，必要时回到 Phase 4 修复

---

## Phase 7: 报告与合规（预计 1-2 天）

> **Exit Criteria**: 
> - 赛题报告 PDF 已生成，覆盖全部 6 项必须内容
> - HONOR_CODE.md、REFERENCE.md 已编写
> - PR 描述已准备，包含全部 7 项必须内容
> - 所有文件交叉引用正确，链接有效

### 7.1 编写赛题报告
文件: `docs/<小组名称>_九齿编译优化_T1-2-1_赛题报告.pdf`

包含：
1. 功能概述与改动范围
2. 技术方案、核心设计和关键代码路径（**含特化的正式正确性论证**：为什么特化是语义等价的）
3. 正确性验证方法与结果（含对抗性场景测试结果）
4. 指标、测试矩阵和对比数据
5. 性能回退、失败用例和不支持场景说明
6. 参考资料、第三方工具和 AI 辅助使用情况

### 7.2 编写 HONOR_CODE.md
内容必须覆盖：
- 独立完成范围的明确声明
- AI 辅助使用情况披露（工具、范围、方式）
- 签名

**建议在 Phase 1 创建初稿**，随项目进展逐步补充。模板结构：
```markdown
# HONOR_CODE.md
## 独立完成范围
## AI 辅助使用
## 外部代码引用
## 签名与日期
```

### 7.3 编写 REFERENCE.md
- 列出所有参考资料（论文、文档、博客）
- 列出参考实现（开源代码、示例）
- 列出外部工具（编译器、分析工具、格式化工具）

### 7.4 PR 描述准备
内容：
1. 赛题编号与小组名称
2. 主要改动点、影响模块和关键代码路径
3. 自测命令、运行环境和结果（**含完整 pytest 输出**）
4. 指标对比表
5. 未覆盖/未实现/已知风险
6. HONOR_CODE.md 和 REFERENCE.md 链接
7. 赛题报告链接

---

## Phase 8: 提交与交叉验证（预计 1 天，7/11 前完成）

> **注意**: 赛题规则注明"提交载体以后续赛题组通知为准"。当前按 GitHub PR 流程准备；若赛题组指定不同提交方式，据此调整。

### 8.1 最终检查
- [ ] `pytest tests/` 全部通过（无跳过、无弱化）
- [ ] `ruff format --check . && ruff check .` 通过
- [ ] 无硬编码尺寸、文件名、benchmark 名
- [ ] 无未声明的外部依赖
- [ ] 所有新增代码有对应的测试覆盖
- [ ] HONOR_CODE.md、REFERENCE.md、赛题报告 PDF 已包含在 PR 中
- [ ] PR 描述包含 Phase 7.4 的全部 7 项内容
- [ ] 分支名符合 kebab-case 规范
- [ ] 提交信息符合 CONTRIBUTING.md 规范

### 8.2 创建 PR
- 分支: `2026-spring-Simon-CHOU-T1-2-1`
- 标题: `[2026春季][T1-2-1] Simon-CHOU`
- 目标: InfiniTensor/ninetoothed main 分支（暂定；以赛题组最终通知为准）

### 8.3 如需要配套 ntops PR
- 在 `D:\ml\ninetooth2026\ntops` 创建配套 PR（如需要新增 benchmark 算子）
- 两侧 PR 交叉引用

### 8.4 PR Review 响应（预留）
- 如维护者在截止前提出修改意见，优先响应
- 预留 1 天（7/12）处理 reviewer 反馈

---

## 风险与注意事项

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| **Windows 上 Triton 不兼容** | 🔴 Critical | Phase 1 预检：若不支持立即切换到 WSL2 或 Linux 远程 GPU 实例 |
| **CUDA GPU 不可用** | 🔴 Critical | Phase 1 确认 `torch.cuda.is_available()`；无 GPU 则申请云 GPU 实例（Lambda Labs/Vast.ai） |
| **隐藏测试 29/30 门槛** | 🔴 Critical | Phase 5.0 对抗性场景设计：每特化 ≥5 个边缘 case；防误命中验证 |
| **Solo 开发无备份** | 🔴 Critical | 识别至少一个可紧急接手的人；代码每日推送到远程分支 |
| 符号 Tensor 的 `is_contiguous` 不可实现 | 🟡 Major | Phase 2.3 原型验证；若不可行则降级为仅 divisible tile 特化 |
| Triton 编译器版本兼容性 | 🟡 Major | 使用与 CI 相同的 triton 版本，固定版本号或 commit hash |
| 代码理解时间超预期 | 🟡 Major | Phase 2 设硬截止（2.5 天）；未完成的 analysis 在实现中补充 |
| 实现后测试发现 bug 无时间修复 | 🟡 Major | Phase 5.5 迭代机制；7/12 全天作为应急缓冲 |
| 浮点误差导致 correctness 不通过 | 🟢 Minor | 保持数值计算路径与 baseline 一致，特化只改变 mask/pointer/stride |
| 性能回退 | 🟢 Minor | 每个 benchmark case 同时测 baseline 和 submitted，确保 speedup ≥ 1.0 |
| 时间不足 | 🟢 Minor | 优先完成 1 个特化类别的完整闭环，再扩展第 2 个；7/12 缓冲日 |
| 赛题提交方式变更 | 🟡 Major | 关注官方通知；PR 结构保持可转换性；不投资提交专用工具 |
| 编译缓存干扰 | 🟢 Minor | 修改生成逻辑后清除 SHA256 缓存（generation.py L873-881），确保新代码生效 |
| 特化误命中 fallback case 扣分 | 🟡 Major | Phase 5.0 防误命中验证；启用条件收紧——宁可漏过也不错杀 |
| ntops 因 ninetoothed 改动而破坏 | 🟡 Major | Phase 4 实现中持续运行 ntops 测试；锁定 ntops commit |

## 时间线概览

```
Phase 1: 环境搭建       ██░░░░░░░░░░  6/25-6/26
Phase 2: Weakness分析    ███░░░░░░░░░  6/27-6/29
Phase 3: 设计选择        ██░░░░░░░░░░  6/30-7/01
Phase 4: 实现           █████░░░░░░░  7/02-7/05
 └ 4.4: 自查                     ●     7/05（0.5天）
Phase 5: 测试           ███░░░░░░░░░  7/05-7/08
 └ 5.0: 对抗性场景               ●     7/05（0.5天优先）
Phase 6: Benchmark      ██░░░░░░░░░░  7/08-7/09
Phase 7: 报告与合规      ██░░░░░░░░░░  7/09-7/10
Phase 8: 提交与验证      █░░░░░░░░░░░  7/11
Buffer (应急缓冲)        █░░░░░░░░░░░  7/12
DEADLINE                                    7/13 0:00 截止 ←
```

**内部截止日（80% 原则）**: 每个 Phase 在估算时间的前 80% 完成主要产出，后 20% 用于收尾和交接。

**迭代路径**: Phase 5 或 Phase 6 发现问题 → 评估修复时间 → 回到 Phase 4 → Phase 4.4 自查 → Phase 5 → Phase 6。最多迭代 2 轮；超出则降级为已知风险并记录在报告中。

---

## 评分对照速查

| 维度 | 分值 | 满分条件 | 门槛 |
|------|------|---------|------|
| Correctness | 30 | 30/30 通过 | **≥29/30**（未达标总分上限 40） |
| Specialization Coverage | 20 | 12/12 命中 + 0 误命中 | 误命中每次扣 1 分 |
| Generated Code Metric | 20 | reduction ≥ 0.25 | reduction ≤ 0 得 0 分 |
| Runtime | 20 | speedup ≥ 1.10 | speedup < 0.95 得 0 分 |
| 工程与报告质量 | 10 | weakness analysis、fallback 设计、代码边界、无硬编码 | — |
