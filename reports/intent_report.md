# FiberMind Intent Classifier Evaluation

- 数据来源：intent_queries_train.jsonl / intent_queries_val.jsonl / intent_queries_test.jsonl (cluster-level split); model=bert-base-chinese; epochs=3; rows(train/val/test)=3466/494/1040; generated_at=2026-09-17T12:25:50.526514+00:00
- 样本数：1040
- 标签数：8

## Overall

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8356 |
| Macro Precision | 0.9224 |
| Macro Recall | 0.8042 |
| Macro F1 | 0.8225 |
| Weighted F1 | 0.8214 |

## Per-class Metrics

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| knowledge_qa | 0.5674 | 0.9837 | 0.7197 | 184 |
| fault_diagnosis | 0.9150 | 1.0000 | 0.9556 | 280 |
| operation_guide | 0.9259 | 0.3333 | 0.4902 | 150 |
| alarm_explanation | 0.9848 | 1.0000 | 0.9924 | 130 |
| history_query | 1.0000 | 1.0000 | 1.0000 | 70 |
| small_talk | 0.9859 | 1.0000 | 0.9929 | 70 |
| unsupported | 1.0000 | 0.5000 | 0.6667 | 70 |
| ambiguous | 1.0000 | 0.6163 | 0.7626 | 86 |

## Confusion Matrix

| true \ pred | knowledge_qa | fault_diagnosis | operation_guide | alarm_explanation | history_query | small_talk | unsupported | ambiguous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knowledge_qa | 181 | 0 | 1 | 2 | 0 | 0 | 0 | 0 |
| fault_diagnosis | 0 | 280 | 0 | 0 | 0 | 0 | 0 | 0 |
| operation_guide | 100 | 0 | 50 | 0 | 0 | 0 | 0 | 0 |
| alarm_explanation | 0 | 0 | 0 | 130 | 0 | 0 | 0 | 0 |
| history_query | 0 | 0 | 0 | 0 | 70 | 0 | 0 | 0 |
| small_talk | 0 | 0 | 0 | 0 | 0 | 70 | 0 | 0 |
| unsupported | 35 | 0 | 0 | 0 | 0 | 0 | 35 | 0 |
| ambiguous | 3 | 26 | 3 | 0 | 0 | 1 | 0 | 53 |

## Notes

指标由 `scripts/train_intent_bert.py` 在独立测试集上真实计算生成；数据按簇切分并通过同标签泄露自检（阈值 0.65）。
