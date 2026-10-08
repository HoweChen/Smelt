# RFC: `smelt doctor` — 用例集与裁判的体检命令

状态：草案（待讨论）
日期：2026-10-08

## 1. 背景

Smelt 的用例大多由 agent 生成。AI 写用例有系统性风险：挑容易的写、断言宽松，
无论 skill 好坏都能通过（"假绿"）。`evaluate_skill` 的分数建立在用例集可信的
前提上，但目前没有任何机制检验这个前提。

本 RFC 引入 `smelt doctor`：**给评测体系本身做体检**，回答"我的尺子准不准"。

命名遵循 `brew doctor` / `flutter doctor` 惯例：不治病，只诊断。

## 2. 定位与分工

| 命令 | 回答的问题 | 包含的检查 |
|---|---|---|
| `smelt doctor` | 尺子准不准 | mutation check（用例灵敏度）、judge canary（裁判校准） |
| `smelt evaluate`（challenge 板块） | skill 好不好 | adversarial probes（对抗探针） |

三者实现上同住 `src/smelt/challenge/` 模块，CLI 表面按"查尺子 / 查 skill"分开。

## 3. Doctor agent

### 3.1 为什么需要专属配置

doctor 的两项检查都需要 LLM：

- **mutation check**：用真实模型重跑用例（脚本回放的后端对变异无感，见 §5.3）。
- **judge canary**：需要一个裁判模型给金丝雀轨迹打分。

复用 `judge` 参数会让职责不清：doctor 测的恰恰包括裁判本身。因此引入独立的
**doctor agent** 配置，框架在运行前显式检查它是否就位。

### 3.2 配置方式：角色化四件套

现状：全局只有一套 `SMELT_API_KEY` / `SMELT_BASE_URL` / `SMELT_LLM_PROVIDER`，
模型按角色分（`SMELT_JUDGE_MODEL`）。凭证和端点却是共享的——无法让 judge 走
OpenAI、doctor 走 Anthropic。

本 RFC 把配置升级为**按角色独立一套**，每个角色四个键：

```
SMELT_<ROLE>_PROVIDER    # openai（默认）| anthropic
SMELT_<ROLE>_BASE_URL    # 可选；省略走 provider 默认端点
SMELT_<ROLE>_API_KEY
SMELT_<ROLE>_MODEL
```

角色清单：`JUDGE`（写作评审 / llm_judge）、`DOCTOR`（本 RFC）、
`AGENT`（被测 agent 自动绑定）、`CHALLENGER`（对抗探针，见 challenge RFC）。

```dotenv
# .env 示例：judge 走 Moonshot，doctor 走 Anthropic，互不影响
SMELT_JUDGE_PROVIDER=openai
SMELT_JUDGE_BASE_URL=https://api.moonshot.cn/v1
SMELT_JUDGE_API_KEY=sk-...
SMELT_JUDGE_MODEL=kimi-k2

SMELT_DOCTOR_PROVIDER=anthropic
SMELT_DOCTOR_API_KEY=sk-ant-...
SMELT_DOCTOR_MODEL=claude-sonnet-4-5
```

回落链（每个键独立回落，显式参数永远优先）：

```
CLI 参数 / API 参数
  > SMELT_<ROLE>_<KEY>
  > SMELT_<KEY>            # 旧的全局键降级为公共兜底，向后兼容
  > provider 默认值 / 报错
```

已有的 `SMELT_API_KEY` / `SMELT_BASE_URL` / `SMELT_LLM_PROVIDER` 保留为公共
兜底，旧配置不破坏。

CLI（四件套均有对应参数）：

```bash
smelt doctor cases.py --skill skills/commit \
    --doctor-provider anthropic --doctor-model claude-sonnet-4-5
# base-url / api-key 一般走 env，也提供 --doctor-base-url / --doctor-api-key
```

Python API：

```python
from smelt import doctor, LLMConfig

# 方式一：显式构造（参数优先，缺省回落 env）
doctor_llm = LLMConfig.from_role("doctor").build()

# 方式二：显式参数覆盖单键
doctor_llm = LLMConfig.from_role("doctor", model="kimi-k2").build()

report = (
    doctor("cases.py", skill="skills/commit", doctor=doctor_llm)
    .run()
)
```

`LLMConfig.from_role(role)` 是统一入口：读四件套 env → 走既有 provider
工厂（`LangChainLLM.from_provider` 的逻辑收编进来）→ 产出 `LLMClient`。
judge / agent / challenger 后续共用同一入口，各自的 env 前缀不同而已。

### 3.3 配置检查（框架职责）

`smelt doctor` 启动时检查 doctor agent 是否配置。**缺失时不静默跳过**——doctor
的全部价值在于验证，静默跳过等于假绿。行为：

```
$ smelt doctor cases.py --skill skills/commit
error: doctor agent not configured.
  set SMELT_DOCTOR_MODEL (+ SMELT_DOCTOR_API_KEY / _BASE_URL / _PROVIDER as needed)
  in .env, or pass --doctor-model / doctor=...
exit 2
```

exit 2（配置错误）与 exit 1（体检发现健康问题）区分，CI 可分别处理。

建议 doctor agent 与被测 agent 用不同模型族，避免同族自我偏好（README 对
judge 已有同样建议）。

## 4. Mutation check

### 4.1 原理

把 SKILL.md 故意改坏一处（变异体），重跑用例集：

- 有用例失败 → 变异体被**杀死**，用例能感知这处改动。
- 全部通过 → 变异体**存活**，没有任何用例守护被改的部分。

```
mutation score = 被杀死的变异体数 / 总变异体数
```

### 4.2 变异算子（确定性文本变换，无需 LLM 生成）

| 算子 | 动作 | 守护它的典型用例 |
|---|---|---|
| `drop_section:<标题>` | 删除一个章节 | 测该章节行为的 case |
| `drop_reference:<路径>` | 不挂载某个 reference | `reference_read` 断言 |
| `drop_constraint` | 删除"禁止/不要"类条款 | `no_tool_call` / 拒答 case |
| `drop_requirement` | 删除"必须/总是"类条款 | 正向行为 case |
| `weaken_trigger` | 模糊化 description 中的触发词 | 触发边界 case |
| `invert_condition` | 反转"如果 A 则 B"条件 | 条件分支 case |

变异在隔离副本上进行，原始 skill 目录不被触碰。

### 4.3 杀死判定（噪声带）

不用裸分差。复用 `compare()` 的显著性判据：

```
杀死 ⟺ 变异后 case 均分下降超过 max(min_delta, 2σ)
```

σ 合并原始与变异两臂的逐次离散度。mutation 重跑默认 `times=1`（粗筛），
边界结果可加跑确认。

### 4.4 `@mutate_check` 装饰器：守护声明

装饰器是**声明**，不是执行——它告诉 doctor 怎么跑，不在 pytest 里触发变异。

```python
from smelt import mutate_check, new_case, text, no_tool_call

# 无注解：默认参与 suite 级杀伤检测（AI 写的裸 case 也逃不掉）
def commit_basic():
    return new_case("commit").given(...).when(text("commit this")).then(tool_call("run_command"))

# 有注解：声明"我这条 case 守护使用边界这一节"
@mutate_check(guards="section:使用边界")
def merge_commit_refusal():
    return (
        new_case("merge-refusal")
        .given(...)
        .when(text("帮我提交这个 merge commit"))
        .then(no_tool_call("run_command"))
    )
```

两级语义：

- **suite 级（无注解）**：某变异体被任一 case 杀死即可，可短路，成本低。
  存活变异体 → 报告并建议补 case。
- **per-case 归属（有注解）**：验证声明的假设——对应变异体必须被**这条 case
  亲手**杀死。杀不死 → 报告"虚假守护"（false guard）。归属检查不能短路，
  只对带注解的 case 做。

实现：装饰器在 import 时把 `(case_name, MutateSpec)` 写入模块级注册表；
`smelt doctor` 收集。不改 `SmeltCase`（frozen dataclass）本身。

`smelt validate` 增加 lint：guards 引用的章节 / reference 必须存在，防止章节
改名后声明失联。

## 5. Judge canary

### 5.1 原理

用 `fixed_agent` 回放一条**故意很差的轨迹**（答非所问 + 编造工具结果），交给
doctor agent 按既有 rubric 打分：

- 得分 < 阈值（默认 0.5）→ 裁判校准正常。
- 得分 ≥ 阈值 → 裁判失准，它此前打的所有分不可信。

### 5.2 内置金丝雀

框架内置一条通用坏轨迹（v1 先内置，自定义留 v2）。判定确定性，成本一次
judge 调用。

### 5.3 降级矩阵

| 条件 | mutation check | judge canary |
|---|---|---|
| doctor agent 未配置 | 不启动（exit 2，§3.3） | 同左 |
| 用例全为确定性后端（ScriptedLLM / fixed_agent） | 标记 `N/A (deterministic backend)` | 正常跑 |
| 无 cases 文件 | 报错（mutation 无对象） | 正常跑 |
| 无 judge 类断言在用例中 | 正常跑 | 提示"用例未使用 llm_judge，canary 仅供参考" |

确定性后端对变异无感（轨迹是脚本回放，不回读 SKILL.md），这是 mutation check
的硬边界，必须显式标记而不是跑出误导性的 0 分。

## 6. 输出

### 6.1 终端

```
$ smelt doctor cases.py --skill skills/commit

Case suite health
  ✔ 12 cases, 6 mutants tested
  ✘ mutation score: 67% (4/6 killed)
    surviving: "drop_section:使用边界" — no case noticed its removal
      → suggestion: add a case asserting merge-commit refusal
    false guard: merge_commit_refusal claims "section:使用边界" but did not kill it

Judge health
  ✔ canary scored 0.2 (threshold 0.5) — judge calibrated

doctor: 1 issue found (mutation score below 0.8)
```

### 6.2 报告文件

写入 `.smelt/reports/doctor/<slug>-<ts>.json|md`（沿用现有报告目录约定），
含变异矩阵明细、逐 case 归属结果、canary 明细。

### 6.3 CI 门

```bash
smelt doctor cases.py --skill skills/commit --min-score 0.8
```

exit 1：mutation score 低于门槛，或 canary 失准。
exit 2：配置错误（doctor agent 缺失等）。
exit 0：健康。

参考门槛：≥0.9 强，0.8～0.89 可进 CI，<0.7 用例集有显著洞（沿用 llm-mutation
的分档经验）。

## 7. 成本护栏

- 变异体默认 5～6 个（每个章节 / reference 一个，加约束类条款）。
- mutation 重跑强制 `times=1`。
- suite 级杀伤短路；归属检查仅限带注解 case。
- 逐次 token 用量汇总进报告（沿用 `trace.metadata["token_usage"]`）。

## 8. 非目标（v1 不做）

- 对抗探针（属于 evaluate 的 challenge 板块，另行设计）。
- 探针 / 存活变异体自动生成 case 代码并落盘（v1 只给建议文本）。
- 自定义金丝雀轨迹。
- mutation score 纳入 `evaluate_skill` 总分。

## 9. 开放问题

1. `guards=` 用章节标题字符串较脆，是否需要 frontmatter 锚点语法？
2. doctor agent 是否允许与被测 agent 同模型（快速模式），还是强制异族？
3. 多 case 文件：`smelt doctor tests/` 递归收集的交互语义。
