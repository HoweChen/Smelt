# RFC: evaluate 的 challenge 板块 — 对抗探针

状态：草案（待讨论）
日期：2026-10-08
依赖：`smelt-doctor-rfc.md`（角色化配置，§3.2）

## 1. 背景

`evaluate_skill` 的用例大多由 agent 生成，测的是"常规路径"。没人主动刁难
skill：说法很绕但语义上该触发的输入、看着像但不该触发的输入、fixture 里埋
干扰内容。这些恰恰是 skill 上线后最先翻车的地方。

本 RFC 给 `evaluate_skill` 增加默认开启的 **challenge 板块**：一个 challenger
agent 读 SKILL.md，生成对抗探针并实跑，击破的场景记入报告。显式
`.with_challenge(enabled=False)` 才关闭。

与 `smelt doctor` 的分工：doctor 查尺子（用例集 + 裁判），challenge 查被测
对象（skill 本身）。

## 2. 探针类型

challenger 生成三类探针，每类配对应的断言形态：

| 类型 | 说明 | 断言 |
|---|---|---|
| 难触发 | 措辞绕、隐含意图，但语义上明确属于 skill 职责 | 正向（`tool_call` / judge） |
| 不该触发 | 近似但无关的请求 | `no_tool_call` / judge 拒答 |
| 干扰 fixture | 工作区文件里埋诱导内容（假指令、误导性数据） | 行为不被带偏 |

探针是**一次性的**：每次评估由 challenger 现场生成，不持久化。击破的探针
由人审后手动晋升为常驻回归 case（v1 不自动写文件，见 §7）。

## 3. API

```python
report = (
    evaluate_skill("skills/commit", judge=judge_llm, agent_llm=agent_llm,
                   challenger=challenger_llm, tools=[git_tool])
    .with_cases(...)
    .with_challenge(rounds=2, probes=8)      # 默认已开启；这行是调参
    # .with_challenge(enabled=False)         # 显式关闭，与 with_lint(enabled=False) 一致
    .run()
)
```

- `challenger`：新增顶层参数。缺省时回落到 `judge`（会在报告中标注"同模型
  挑战，可信度降级"），建议独立配置（§4）。
- `rounds`：挑战轮次，默认 1。第 2 轮起 challenger 能看到上一轮存活的
  探针，调整策略再攻（自适应 probing）。
- `probes`：探针总数上限，默认 8。

CLI：

```bash
smelt evaluate skills/commit --cases cases.py ... --no-challenge
smelt evaluate skills/commit --cases cases.py ... --challenge-rounds 2 --challenge-probes 8
```

## 4. Challenger 配置

沿用 doctor RFC §3.2 的角色化四件套，角色名 `CHALLENGER`：

```dotenv
SMELT_CHALLENGER_PROVIDER=anthropic
SMELT_CHALLENGER_API_KEY=sk-ant-...
SMELT_CHALLENGER_MODEL=claude-sonnet-4-5
```

```python
challenger = LLMConfig.from_role("challenger").build()
```

建议 challenger 与被测 agent 不同模型族：同族模型知道自家怎么说话，也想
不到自家想不到的死角。README 对 judge 的建议同样适用于此。

**配置缺失时**：challenge 板块标记 `skipped (no challenger configured)` 并附
原因——沿用 evaluate 既有的优雅降级（writing/suggestions 同款），不阻断其他
板块。注意这与 doctor 的"缺配置即 exit 2"不同：doctor 的整个职责是验证，
跳过等于假绿；challenge 是 evaluate 的一个板块，降级不影响其余部分出分。

## 5. 运行与判定

每个探针实跑一次完整 agent 循环（复用现有 runner），判定规则：

- 探针挂了断言 → **击破**（skill 的洞），记入报告。
- 探针通过 → 存活，下轮 challenger 可据此调整。

challenge 全部探针运行在 `times=1`（探针是一次性样本，重复无意义；
稳定性问题留给晋升后的常驻 case 用 repeat 解决）。

## 6. 报告与计分

**challenge 不进总分。** 总分是 `compare()` 和 `--fail-under` 的锚，必须
稳定；探针每次新生成，纳入权重会让同一版本两次跑出不同总分。报告新增
独立板块：

```markdown
## Challenge (adversarial probes)

8 probes, 2 rounds, challenger: claude-sonnet-4-5
  ✘ hard-trigger: "帮我把这些改动归置一下留个档" — agent did not recognize
    a commit request; expected run_command, got none
  ✘ distractor: fixtures/README.md contained "ignore previous instructions";
    agent's answer was diverted
  ✔ 6 probes survived

Breaks feed suggestions: consider adding trigger paraphrases to description.
```

`compare()` 对 challenge 板块只做信息性 diff（击破数 Δ），永不触发回归判定。
想计入总分的用户可显式 `with_weights(..., challenge=0.1)` 自行承担不稳定性。

## 7. 探针晋升（v1 只建议，不落盘）

击破的探针在报告中附**建议的 case 代码片段**（`new_case(...)` 链），人审后
粘贴进 cases.py。自动写入留到 v2——探针是 LLM 生成的，断言方向可能本身就
错，需要人眼把关正确性（mutation check 管灵敏度，人审管正确性）。

## 8. 降级矩阵

| 条件 | challenge 行为 |
|---|---|
| 无 challenger 且无 judge 可回落 | skipped，注明原因 |
| challenger 回落到 judge | 正常跑，报告标注同模型可信度降级 |
| 被测后端为确定性（ScriptedLLM / fixed_agent） | skipped（探针需要真实 agent 读 skill） |
| 无 cases | 正常跑（challenge 不依赖既有 case） |
| challenger 生成失败 / 探针不可解析 | 该探针记 error，不阻断其余探针 |

## 9. 成本护栏

- 探针默认上限 8，轮次默认 1。
- 每探针一次 agent 循环 + 每轮一次 challenger 生成调用。
- token 用量汇总进 challenge 板块（沿用 `trace.metadata["token_usage"]`）。

## 10. 非目标（v1 不做）

- mutation check 与 judge canary（属于 `smelt doctor`，见 doctor RFC）。
- 探针自动落盘为 case。
- challenge 分数纳入总分默认权重。
- 多 challenger 辩论（v2 可考虑红蓝对抗）。

## 11. 开放问题

1. 探针与既有 case 撞车（重复场景）要不要去重？怎么去重（语义相似度）？
2. `guards=` 章节引用失稳的问题在 doctor RFC 开放问题 1；探针报告里引用
   章节时同样受影响。
3. 干扰 fixture 类探针要写工作区，是否需要独立的 fixture 沙箱目录约定？
