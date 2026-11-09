"""Deterministic intent-query dataset generator with diversity transforms.

与旧版“固定模板 + 同义词槽填充”不同，本生成器：
  1. 每个意图由数十个语义簇种子（scenario）构成，每个种子是一个簇（cluster），
     簇内样本再叠加随机多样性变换，避免句式复制；
  2. 变换覆盖真实运维表达：口语化、错别字、简写、不完整 Query、模糊指代、
     多意图边界、Hard Negative、同词不同意图；
  3. train/val/test 以“簇”为单位切分（cluster-aware split），同簇样本永远同侧，
     从机制上避免近重复样本跨集造成的数据泄露；切分后运行泄露自检。

该脚本完全离线、确定性（固定 seed），用于 CI 与无 API Key 环境。
需要更高多样性的真实语料时请使用 scripts/generate_intent_dataset_llm.py。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from app.evaluation.dataset_split import (
    split_with_leakage_guard,
)

ROOT = Path(__file__).resolve().parent.parent

DEVICES = ["ONU", "ONT", "OLT", "PON口", "光模块", "光纤链路", "传输设备", "ODF架", "光猫"]
ALARMS = ["LOS", "LOF", "OOF", "CRC_ERROR", "BER_HIGH", "DYING_GASP", "AUTH_FAIL"]
SYMPTOMS = ["频繁掉线", "红灯闪烁", "业务中断", "误码升高", "无法注册", "网速变慢", "反复重启"]
SCENES = ["从昨晚开始", "割接之后", "下雨以后", "早高峰", "重启之后", "最近一周", "值班时"]

# ---------------------------------------------------------------------------
# 语义簇种子：{intent: [(seed_sentence with slots), ...]}
# 每个种子代表一种独立的表达场景；同一种子生成的样本属于同一簇。
# ---------------------------------------------------------------------------
SEEDS: dict[str, list[str]] = {
    "fault_diagnosis": [
        "{scene}{device}一直{symptom}，还报{alarm}，应该怎么排查",
        "现场发现{device}{symptom}并出现{alarm}告警，可能是什么原因",
        "{device}收光功率-{power} dBm而且{symptom}，如何定位",
        "同一机房的{device}{scene}{symptom}，先检查什么",
        "{device}全部{symptom}，网管上{alarm}刷屏，怎么快速定位是单点还是共性故障",
        "{device}白天正常一到{scene}就{symptom}，伴生{alarm}，怀疑哪里的问题",
        "新装的{device}激活不了，注册上去就掉，报{alarm}，光衰-{power}，帮我看看",
        "{device}{symptom}，{alarm}告警反复上下线，是不是光路的问题",
        "割接完{device}就开始{symptom}，{alarm}间歇性上报，从哪儿开始查",
        "{device}只有几台{symptom}其他都正常，还报{alarm}，是不是尾纤问题",
        "老小区那片{device}{scene}{symptom}，{alarm}告警集中爆发，怎么排查",
        "{device}收光-{power}还{symptom}，{alarm}和掉线有没有关系",
        "帮忙定位下：{scene}{device}批量{symptom}，网管全是{alarm}",
        "{device}{symptom}三天了，{alarm}一直清不掉，现场看光缆没问题",
        "{alarm}告警导致{device}{symptom}，该怎么一步步定位根因",
    ],
    "knowledge_qa": [
        "{device}正常的接收光功率范围是多少",
        "{alarm}和光功率有什么关系",
        "为什么{device}会出现{symptom}",
        "介绍一下{device}的日常维护规范",
        "{device}的发光功率标准值一般是多少",
        "{alarm}告警的触发条件是什么",
        "{device}和OLT之间衰减怎么算",
        "光链路预算怎么评估，{device}能承受多大衰减",
        "{device}{symptom}一般有哪些常见原因，给我讲讲原理",
        "{alarm}的严重等级是怎么定义的",
        "{device}注册流程是什么，正常要多久",
        "PON口下挂的{device}数量上限一般是多少",
        "{device}的寿命和光功率衰减有什么关系",
        "冷备份的{device}怎么验收光指标",
    ],
    "operation_guide": [
        "如何清洁{device}的光纤连接器",
        "更换{device}前需要做哪些操作",
        "请给出处理{alarm}的标准步骤",
        "怎么用光功率计检查{device}",
        "{device}怎么登录开局配置",
        "怎么给{device}做光功率测试并记录",
        "{device}更换的完整流程是什么，需要注意什么",
        "如何批量导出{device}的{alarm}告警清单",
        "{device}重启操作规范是什么，要不要先切业务",
        "怎么把{device}从PON口下线再重新注册",
        "日常巡检{device}要检查哪些项目",
        "{alarm}告警确认和清除的操作步骤",
        "{device}尾纤怎么熔接，衰耗控制多少合适",
    ],
    "alarm_explanation": [
        "{alarm}是什么意思",
        "网管上的{alarm}告警代表什么",
        "请解释{device}的{alarm}告警",
        "{alarm}和LOS有什么区别",
        "{alarm}这个告警严重吗，影响什么业务",
        "{alarm}是什么原因触发的",
        "{alarm}告警含义是什么，跟光衰有关系吗",
        "帮我说下{alarm}是什么告警，什么时候会上报",
        "{device}面板灯快闪和{alarm}是一回事吗",
        "{alarm}具体指什么故障，和{symptom}有关吗",
    ],
    "history_query": [
        "我上次问的{device}问题是什么",
        "把之前的诊断记录找出来",
        "刚才关于{alarm}你给了什么建议",
        "查看这个会话的历史记录",
        "我之前问过光功率的事吗",
        "刚刚你说的排查步骤再发一遍",
        "这轮对话我们都聊了什么",
        "上次那个{symptom}的结论是什么来着",
        "回顾一下刚才的处理建议",
        "我今天都问过哪些告警",
    ],
    "small_talk": [
        "你好，我想咨询一下网络问题",
        "谢谢你的帮助",
        "你是谁",
        "今天辛苦了，再见",
        "在吗，有人吗",
        "你能做什么",
        "中午好呀",
        "明白了，谢谢你啊",
        "你叫什么名字",
        "好的知道了",
    ],
    "unsupported": [
        "帮我写一首关于{device}的诗",
        "查询今天的天气",
        "给我推荐一部电影",
        "计算{power}乘以二是多少",
        "今天股票行情怎么样",
        "给我讲个笑话",
        "帮我订个外卖",
        "写一段辞职信",
        "明天适合爬山吗",
        "帮我抢个演唱会门票",
    ],
    "ambiguous": [
        "这个怎么办",
        "不行了",
        "帮我看下",
        "它又坏了",
        "这个告警",
        "还是不行",
        "那玩意又出问题了",
        "刚刚那个又来了",
        "{device}",
        "一直闪红灯",
        "怎么办",
        "上不去",
        "又出现了",
        "还是老问题",
    ],
}

TARGET_COUNTS = {
    "fault_diagnosis": 1400,
    "knowledge_qa": 850,
    "operation_guide": 650,
    "alarm_explanation": 650,
    "history_query": 350,
    "small_talk": 350,
    "unsupported": 350,
    "ambiguous": 400,
}

# ---------------------------------------------------------------------------
# 多样性变换器：每条样本随机叠加 0-2 个变换，形成同簇内的表达差异
# ---------------------------------------------------------------------------
TYPO_MAP = {
    "接收": "接搜",
    "掉线": "调线",
    "功率": "公率",
    "告警": "靠警",
    "闪断": "闪段",
    "光衰": "光捎",
    "中断": "中段",
    "误码": "误马",
    "注册": "注测",
    "更换": "更焕",
    "清洁": "清结",
    "定位": "定味",
}

SPOKEN_PREFIXES = ["师傅，", "麻烦问下，", "急，", "现场等的，", "哥，", "那个，"]
SPOKEN_SUFFIXES = ["呗", "哈，谢了", "，急等", "，谢谢", "，在线等", "，快点呗"]
VAGUE_REPLACEMENTS = ["它", "这个", "那个", "这台", "这玩意"]

ABBREVIATIONS = {
    "接收光功率": "收光",
    "PON口": "P口",
    " dBm": "",
    "光网络终端": "光猫",
    "光线路终端": "OLT",
}


def apply_typo(text: str, rng: random.Random) -> str:
    candidates = [source for source in TYPO_MAP if source in text]
    if not candidates:
        return text
    source = rng.choice(candidates)
    return text.replace(source, TYPO_MAP[source], 1)


def apply_spoken(text: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        text = rng.choice(SPOKEN_PREFIXES) + text
    if rng.random() < 0.5:
        text = text + rng.choice(SPOKEN_SUFFIXES)
    return text


def apply_abbreviation(text: str, rng: random.Random) -> str:
    candidates = [source for source in ABBREVIATIONS if source in text]
    if not candidates:
        return text
    source = rng.choice(candidates)
    return text.replace(source, ABBREVIATIONS[source], 1)


def apply_vague_reference(text: str, rng: random.Random, devices: list[str]) -> str:
    if not devices:
        return text
    device = rng.choice(devices)
    if device in text:
        return text.replace(device, rng.choice(VAGUE_REPLACEMENTS), 1)
    return text


def apply_ellipsis(text: str, rng: random.Random) -> str:
    # 不完整 Query：按标点截断尾部子句
    for separator in ("，", "，还", "，而且", "，同时"):
        position = text.rfind(separator)
        if position > 4:
            return text[:position]
    return text


TRANSFORMS = ["typo", "spoken", "abbreviation", "ellipsis"]


def build_sample(
    label: str, seed_index: int, seed_text: str, index: int, rng: random.Random
) -> dict:
    device = rng.choice(DEVICES)
    alarm = rng.choice(ALARMS)
    symptom = rng.choice(SYMPTOMS)
    scene = rng.choice(SCENES)
    power = rng.randint(7, 35)
    text = seed_text.format(
        device=device, alarm=alarm, symptom=symptom, scene=scene, power=power
    )

    # 同簇样本差异 = 随机叠加 0-2 个变换（ambiguous/small_talk 只做轻变换保持原语义）
    heavy_transforms_allowed = label not in {"small_talk", "ambiguous"}
    chosen: list[str] = []
    if heavy_transforms_allowed:
        chosen = rng.sample(TRANSFORMS, k=rng.randint(0, 2))
    elif rng.random() < 0.4:
        chosen = ["spoken"]
    for transform in chosen:
        if transform == "typo":
            text = apply_typo(text, rng)
        elif transform == "spoken":
            text = apply_spoken(text, rng)
        elif transform == "abbreviation":
            text = apply_abbreviation(text, rng)
        elif transform == "ellipsis":
            text = apply_ellipsis(text, rng)
    if heavy_transforms_allowed and rng.random() < 0.12:
        # 少量样本把设备名替换为模糊指代，模拟多轮/省略主语的真实提问
        text = apply_vague_reference(text, rng, [device])

    entities: dict[str, object] = {}
    if device in text:
        entities["device_type"] = device
    if alarm in text.upper():
        entities["alarm_codes"] = [alarm]
    if symptom in text:
        entities["symptoms"] = [symptom]
    return {
        "id": f"intent-{index:05d}",
        "text": text,
        "intent": label,
        "slots": entities,
        # 簇 ID：同一种子 + 同意图为一个簇，切分以簇为单位防止数据泄露
        "cluster_id": f"{label}:seed-{seed_index:03d}",
        "transforms": chosen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a diverse 5,000-query intent dataset with cluster-aware splits"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "generated" / "intent_queries.jsonl"
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.1, 0.2),
        metavar=("TRAIN", "VAL", "TEST"),
        help="cluster-level split ratios",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    samples: list[dict] = []
    index = 1
    for label, count in TARGET_COUNTS.items():
        seeds = SEEDS[label]
        per_seed = count // len(seeds)
        remainder = count - per_seed * len(seeds)
        # 按簇分配数量：每个种子承担 per_seed 条，簇内继续随机变换形成差异
        for seed_index, seed_text in enumerate(seeds):
            for _ in range(per_seed):
                samples.append(build_sample(label, seed_index, seed_text, index, rng))
                index += 1
        # 余量补到随机种子（保持总量与 TARGET_COUNTS 接近）
        for _ in range(remainder):
            seed_index = rng.randrange(len(seeds))
            samples.append(build_sample(label, seed_index, seeds[seed_index], index, rng))
            index += 1

    rng.shuffle(samples)

    # 簇级切分 + 泄露守护：不仅同构造簇同侧，还会把“不同种子但语义重叠”
    # 引发的跨侧高相似对迭代合并，直到没有泄露。
    cluster_of_row, cluster_split, residual_leaks = split_with_leakage_guard(
        samples,
        ratios=tuple(args.ratios),
        seed=args.seed,
    )
    for sample, cluster_id in zip(samples, cluster_of_row, strict=True):
        sample["cluster_id"] = cluster_id
        sample["split"] = cluster_split[cluster_id]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    for split in ("train", "val", "test"):
        path = args.output.with_name(f"{args.output.stem}_{split}.jsonl")
        with path.open("w", encoding="utf-8") as file:
            for sample in samples:
                if sample["split"] == split:
                    file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # ---- 自检报告：多样性 + 防泄露 ----
    texts = [sample["text"] for sample in samples]
    unique_ratio = len(set(texts)) / len(texts)
    split_counts = Counter(sample["split"] for sample in samples)
    cluster_count = len({sample["cluster_id"] for sample in samples})

    print(f"wrote {len(samples)} rows to {args.output}")
    print(f"clusters: {cluster_count} | unique texts: {len(set(texts))}/{len(texts)} "
          f"({unique_ratio:.1%})")
    print("splits:", json.dumps(dict(split_counts), ensure_ascii=False))
    per_split_intents = {
        split: dict(Counter(s["intent"] for s in samples if s["split"] == split))
        for split in ("train", "val", "test")
    }
    for split, intents in per_split_intents.items():
        print(f"intents[{split}]:", json.dumps(intents, ensure_ascii=False))
    if residual_leaks:
        print(f"WARNING: {len(residual_leaks)} cross-split pairs remain after guard")
        for leak in residual_leaks[:5]:
            print("  ", leak)
    else:
        print("leakage check: no same-label cross-split pair above similarity threshold")


if __name__ == "__main__":
    main()
