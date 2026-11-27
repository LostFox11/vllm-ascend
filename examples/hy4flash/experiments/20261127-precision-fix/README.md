# HY4 precision fix validation — 2026-11-27

已定位的 Target Attention、MoE、IndexPool 精度问题已修复。
本次向 `LostFox11/vllm-ascend:Hy4-flash-opti` 提交修复供审阅，**暂不合入**。

## 服务启动与测评方法

使用仓库正常启动服务的方式：`examples/hy4flash/pd_118/start_prefill.sh`、
`start_decode.sh` 和 `bin/vllm`。部署时仅适配容器路径和上下文长度；
使用模型官方 prompt/chat template，通过 EvalScope 1.15.0 官方 `aime26`
adapter 测 AIME2026 全部 30 题。

- 服务上下文：128K（`max_model_len=131072`）。该参数是上下文容量。
- 并发：16（`eval_batch_size=16`）。
- 后处理/采样参数：`temperature=0.9`、`top_p=1`、`top_k=-1`。
- 没有额外指定输出长度、生成 seed 或 repeats。
- P：物理卡 0–3，TP4 eager；D：物理卡 4–7，TP1/DP4/EP4。
  保留 DFLY5、`hy4flash_dfly_dp_graph=true` 和正常图尺寸配置。
- P 与全部 D 进程均加载本轮候选：`363c1616b` 加本 PR 的精度修复。
  模型服务容器为 `hy4-pr22839-363c1616b-1127`，源码目录为
  `/opt/hy4/validation/fia-moe-precision-20261127/source`。

EvalScope 的核心配置为：

```python
TaskConfig(
    model="hy4-flash",
    api_url="http://<service-proxy>:<port>/v1",
    datasets=["aime26"],
    eval_batch_size=16,
    generation_config={"temperature": 0.9, "top_p": 1, "top_k": -1},
)
```

本轮通过官方 `local_path` 接口加载原始 30 题缓存。原始数据版本为
`evalscope/aime26@12ae93b463353acd4d272dbbc7dbecbcfb902642`。
为适配 EvalScope，缓存仅将答案整数类型转为字符串，题目和答案数值不变。
长题触发 HTTP 超时后使用官方缓存续跑；仅调整 timeout/stream 等传输参数，
未改变 prompt、采样参数、评分器或模型服务代码。

## 已有结果

截至保存的阶段快照：**完成 29/30 题，官方评分 29/29 正确**。
最后一题为零基 index=14（第 15 题），本次提交不等待其最终结果。
这是已完成题目的阶段结果，不是最终 30 题测评分数。
`aime2026-partial-results.json` 保存逐题官方 acc、题号与原评分文件 SHA256。

最后一题已观察到持续生成和同一思路反复验证，尚未收敛；不能把它记为正确、
错误或服务卡死。已完成 29 题没有检出机械重复的启发式标记；这不代替人工证明。

## 修复范围与其他验证

- Attention：保留 FP32 C8 KV 与 QK，用融合 sink/softmax/PV 替换不满足这些
  数值边界的 CANN MLA FIA 路径；最终 Attention tile 为 64。
- MoE：恢复 BF16 expert sum 中间舍入，再执行 CANN AddRMSNorm。
- IndexPool：恢复 key RoPE，补齐短上下文合法 dense 候选，padding 保持无效。
- IndexPool 新增 11 项回归修复前全部失败、修复后通过；联合测试记录
  `73 passed, 72 skipped`，跳过项不计作 NPU 检查。
- 修复过程中的 NPU 组件记录为 34 项通过；该记录对应较早组件版本，
  不能当作最终候选全部文件的独立复测。最终候选已有上述正常服务 AIME 结果。
- 修复包应用校验、候选源码一致性、Python 语法及 Ruff 记录见
  [交付证据](https://github.com/peterDengcx/vllm-ascend-118/tree/36171c25152120fff3c6cdcfc484610434e145d6/examples/hy4flash/experiments/20261125-verify-fusions/precision-fix-20261127)。
- 本地 `bash format.sh ci` 因缺少 pre-commit 未能执行完整全仓检查。
  没有完成本候选的性能验收；本 PR 不宣称吞吐提升。

Engram rstd 与 iHC head 的剩余数值差异不在本 PR 修复范围内。
