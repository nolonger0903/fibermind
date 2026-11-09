# FiberMind Offline Evaluation

> Proxy metrics are deterministic smoke-test metrics. Use `run_ragas.py` for model-based RAGAS scores.

| Metric | Mean |
| --- | ---: |
| faithfulness_proxy | 1.0 |
| answer_relevancy_proxy | 0.75 |
| context_recall | 0.8333 |
| context_relevancy_proxy | 0.8333 |
| latency_ms | 2.8167 |

## Cases

- `1.00` recall / `6.0 ms`: ONU频繁闪断，收光功率-29 dBm，并出现LOS告警，应该怎么排查？
- `1.00` recall / `2.8 ms`: 同一个PON口下面十几台ONU同时离线，OLT报PON_LOS
- `1.00` recall / `3.8 ms`: 链路CRC持续增长并且误码率很高，要检查哪些指标？
- `1.00` recall / `2.8 ms`: LOF告警但没有LOS告警，先排查哪里？
- `0.00` recall / `0.1 ms`: ONU有光但认证失败，序列号可能不一致
- `1.00` recall / `1.4 ms`: 短距离光模块接收光太强并有误码怎么办？