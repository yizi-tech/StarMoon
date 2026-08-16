#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_tokenizer_corpus.py
===========================
为 StarMoon-z1 分词器训练，从公开语料库**流式采样**已清洗文本并落盘为 JSONL。

设计依据: docs/tokenizer-design.md §3 语料配比。

特性:
- 流式加载 (datasets streaming)：不下载全量 TB 级语料，只按需采样到目标体量。
- 默认走 HF 国内镜像 (hf-mirror.com)，适配 AutoDL 等国内服务器。
- 按权重分配各源字节预算；权重在 SOURCES 中给出，运行时自动归一化。
- 多模态源自动注入 <image>/<video>/<audio> 占位符，让分词器见过这些特殊 token
  （OBELICS 文本本身已含 <image>，其余多模态源按轮换注入）。
- 输出: <output_dir>/raw/corpus.jsonl  +  <output_dir>/manifest.json（含来源/license/计数）。

用法 (AutoDL):
    pip install datasets
    # HF_ENDPOINT 脚本已默认设为 hf-mirror.com，可 --hf-endpoint "" 关闭
    python scripts/download_tokenizer_corpus.py --output-dir /root/data/tok_corpus --target-gb 15

说明:
- 只下载/抽取**文本**，不含图像/视频/音频文件（分词器训练只需要文本侧）。
- 仅采样，不做深度清洗；选用源本身已是清洗/去重过的（见 docs/tokenizer-design.md）。
- 许可证: CCI / ShareGPT4V 等为 NC-ND 或非商用，仅作研究；要进可商用分词器请
  在 SOURCES 中调低其权重或剔除。
"""
import os
import sys
import json
import hashlib
import logging
import argparse
import random

try:
    from tqdm import tqdm
except ImportError:  # tqdm 可选
    def tqdm(x, **kw):
        return x

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TokCorpus")

# ──────────────────────────────────────────────
# 语料源注册表
#   w        : 相对权重（运行时归一化，不必严格和为 1）
#   fields   : 候选文本字段，按顺序取第一个非空字符串
#   cat      : 类别（仅用于 manifest 归类）
#   lic      : 许可证（务必自查后合规使用）
#   trc      : 是否需要 trust_remote_code（部分 HF 仓库的加载脚本）
# 配比概览（归一化后）: 中文 ~35% / 英文 ~15% / 代码 ~20% / 数学 ~15% / 多模态 ~5%
#   学科/领域(domain) 已折叠进 WanJuan(FineWeb 亦含专业域)，不单列以免重复计数。
# ──────────────────────────────────────────────
SOURCES = [
    dict(name="CCI3-HQ",        repo="BAAI/CCI3-HQ",                     fields=["text"],            cat="chinese",   lic="CC-BY-NC-ND-4.0",          w=0.12, trc=True),
    dict(name="Chinese-FineWeb-Edu", repo="opencsg/chinese-fineweb-edu-v2", fields=["text"],       cat="chinese",   lic="OpenCSG社区许可(商用需申请)", w=0.08, trc=False),
    dict(name="WanJuan-Text",   repo="Shanghai_AI_Laboratory/Wanjuan-1.0-Text", fields=["content", "text"], cat="chinese", lic="CC-BY-4.0", w=0.15, trc=True),
    dict(name="FineWeb-Edu",    repo="HuggingFaceFW/fineweb-edu",       fields=["text"],            cat="english",   lic="ODC-BY-1.0",              w=0.10, trc=False),
    dict(name="DCLM",           repo="mlfoundations/dclm-baseline-1.0",  fields=["text"],            cat="english",   lic="ODC-BY",                  w=0.05, trc=False),
    dict(name="TheStack-v2",    repo="bigcode/the-stack-v2",             fields=["content", "text"], cat="code",     lic="逐文件许可(OpenRAIL等)",   w=0.20, trc=False),
    dict(name="MegaMath",       repo="LLM360/MegaMath",                  fields=["text"],            cat="math",      lic="见仓库",                  w=0.15, trc=False),
    dict(name="OBELICS",        repo="HuggingFaceM4/OBELICS",            fields=["text"],            cat="multimodal", lic="见仓库",                  w=0.03, trc=True),
    dict(name="ShareGPT4V",     repo="Lin-Chen/ShareGPT4V",              fields=["caption", "text", "value"], cat="multimodal", lic="CC-BY-NC-4.0", w=0.01, trc=True),
    dict(name="WenetSpeech",    repo="wenet/WenetSpeech",                fields=["txt", "text"],     cat="multimodal", lic="Apache-2.0",             w=0.01, trc=True),
]

PLACEHOLDERS = ["<image>", "<video>", "<audio>"]


def load_stream(repo: str, trc: bool):
    """加载流式数据集，兼容无显式 train split 的仓库。"""
    try:
        return load_dataset(repo, split="train", streaming=True, trust_remote_code=trc)
    except Exception:
        ds = load_dataset(repo, streaming=True, trust_remote_code=trc)
        key = list(ds.keys())[0]
        logger.warning(f"  {repo} 无 train split，改用 '{key}'")
        return ds[key]


def extract_text(example: dict, fields) -> str | None:
    """按顺序从候选字段取第一个可用文本；均无则扫描首个够长的字符串值。"""
    for f in fields:
        v = example.get(f)
        if isinstance(v, str) and v.strip():
            return v
    for v in example.values():
        if isinstance(v, str) and len(v.strip()) > 20:
            return v
    return None


def maybe_inject(text: str, cat: str, rng: random.Random) -> str:
    """多模态源：若文本中尚无占位符，则轮换注入一个，让分词器见过三类 token。"""
    if cat == "multimodal":
        if not any(p in text for p in PLACEHOLDERS):
            ph = PLACEHOLDERS[rng.randint(0, len(PLACEHOLDERS) - 1)]
            text = ph + "\n" + text
    return text


def sample_source(src: dict, target_bytes: int, args, rng: random.Random, seen: set):
    """流式采样单个源，达到字节预算或文档上限即停。返回 (docs, bytes)。"""
    try:
        ds = load_stream(src["repo"], src["trc"])
    except Exception as e:
        logger.warning(f"  [{src['name']}] 加载失败，跳过: {e}")
        return 0, 0

    n_written, b_written = 0, 0
    pbar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc=src["name"])
    for ex in ds:
        if b_written >= target_bytes or n_written >= args.max_docs_per_source:
            break
        text = extract_text(ex, src["fields"])
        if not text or len(text) < args.min_chars:
            continue
        # 轻量去重（仅本次运行内精确去重，控制内存）
        h = hashlib.md5(text[:200].encode("utf-8")).hexdigest()
        if h in seen:
            continue
        if len(seen) < args.dedup_cache:
            seen.add(h)
        text = maybe_inject(text, src["cat"], rng)
        yield text
        nb = len(text.encode("utf-8"))
        b_written += nb
        n_written += 1
        pbar.update(nb)
    pbar.close()
    return n_written, b_written


def parse_args():
    p = argparse.ArgumentParser(description="StarMoon-z1 分词器语料流式采样")
    p.add_argument("--output-dir", default="./tokenizer_corpus", help="输出目录")
    p.add_argument("--target-gb", type=float, default=15.0, help="目标原始文本体量(GB)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-endpoint", default="https://hf-mirror.com",
                   help="HuggingFace 镜像；传空字符串 '' 关闭(直连 hf.co)")
    p.add_argument("--min-chars", type=int, default=50, help="单条文本最小字符数")
    p.add_argument("--max-docs-per-source", type=int, default=5_000_000, help="单源文档上限(安全阀)")
    p.add_argument("--dedup-cache", type=int, default=2_000_000, help="去重哈希集合容量上限")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的 corpus.jsonl")
    p.add_argument("--no-trust-remote-code", action="store_true",
                   help="禁用所有源的 trust_remote_code（部分仓库可能因此加载失败）")
    p.add_argument("--dry-run", action="store_true", help="只打印采样计划后退出")
    return p.parse_args()


def main():
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        logger.info(f"HF_ENDPOINT={args.hf_endpoint}")

    if args.no_trust_remote_code:
        for s in SOURCES:
            s["trc"] = False

    total_budget = int(args.target_gb * 1e9)
    total_w = sum(s["w"] for s in SOURCES)

    # dry-run: 打印计划
    if args.dry_run:
        logger.info("=== 采样计划 (dry-run) ===")
        for s in SOURCES:
            bud = total_budget * s["w"] / total_w
            logger.info(f"  {s['name']:<22} cat={s['cat']:<10} ~{bud/1e9:.2f}GB  lic={s['lic']}")
        logger.info(f"  合计目标: {args.target_gb:.1f}GB")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    raw_dir = os.path.join(args.output_dir, "raw")
    raw_path = os.path.join(raw_dir, "corpus.jsonl")
    os.makedirs(raw_dir, exist_ok=True)
    if os.path.exists(raw_path) and not args.overwrite:
        logger.error(f"{raw_path} 已存在；加 --overwrite 覆盖，或换 --output-dir")
        return

    rng = random.Random(args.seed)
    manifest = {"target_gb": args.target_gb, "hf_endpoint": args.hf_endpoint, "sources": {}}
    seen: set = set()

    with open(raw_path, "w", encoding="utf-8") as fout:
        for src in SOURCES:
            budget = int(total_budget * src["w"] / total_w)
            logger.info(f"=== 采样 {src['name']} (cat={src['cat']}, ~{budget/1e9:.2f}GB, lic={src['lic']}) ===")
            n, b = 0, 0
            for text in sample_source(src, budget, args, rng, seen):
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n += 1
                b += len(text.encode("utf-8"))
            manifest["sources"][src["name"]] = {
                "cat": src["cat"], "license": src["lic"], "docs": n, "bytes": b,
            }
            logger.info(f"    -> {n:,} docs, {b/1e9:.3f} GB")

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fm:
        json.dump(manifest, fm, ensure_ascii=False, indent=2)

    total_docs = sum(v["docs"] for v in manifest["sources"].values())
    total_bytes = sum(v["bytes"] for v in manifest["sources"].values())
    logger.info(f"完成: {total_docs:,} docs, {total_bytes/1e9:.2f} GB")
    logger.info(f"  语料: {raw_path}")
    logger.info(f"  清单: {manifest_path}")


if __name__ == "__main__":
    main()
