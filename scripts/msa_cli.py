#!/usr/bin/env python
"""
MSA 命令行工具
==============

子命令：
  encode    编码 jsonl 语料为记忆库
  query     加载记忆库并查询（单轮 / Interleave 多轮）
  add       向已有记忆库增量添加文档（jsonl）

示例：
  # 1) 编码语料
  python scripts/msa_cli.py encode \
      --model_path ./ckpt/StarMoon-z1-7B-MSA \
      --corpus ./data/corpus.jsonl \
      --output ./memory_cache/my_memory.pt

  # 2) 查询
  python scripts/msa_cli.py query \
      --model_path ./ckpt/StarMoon-z1-7B-MSA \
      --memory ./memory_cache/my_memory.pt \
      --prompt "StarMoon-z1 是什么？" \
      --top_k 3 --interleave

  # 3) 增量添加
  python scripts/msa_cli.py add \
      --model_path ./ckpt/StarMoon-z1-7B-MSA \
      --memory ./memory_cache/my_memory.pt \
      --corpus ./data/new_docs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="StarMoon-z1 MSA 工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="编码 jsonl 语料为记忆库")
    pe.add_argument("--model_path", required=True, help="StarMoon-z1-MSA 模型目录")
    pe.add_argument("--corpus", required=True, help="jsonl 语料路径")
    pe.add_argument("--output", required=True, help="记忆库输出路径 (.pt)")
    pe.add_argument("--batch_size", type=int, default=8)
    pe.add_argument("--device", default="cuda")

    pq = sub.add_parser("query", help="加载记忆库并查询")
    pq.add_argument("--model_path", required=True)
    pq.add_argument("--memory", required=True, help="记忆库路径 (.pt)")
    pq.add_argument("--prompt", required=True)
    pq.add_argument("--top_k", type=int, default=5)
    pq.add_argument("--max_new_tokens", type=int, default=512)
    pq.add_argument("--interleave", action="store_true", help="启用 Memory Interleave 多轮")
    pq.add_argument("--device", default="cuda")

    pa = sub.add_parser("add", help="增量添加文档到记忆库")
    pa.add_argument("--model_path", required=True)
    pa.add_argument("--memory", required=True)
    pa.add_argument("--corpus", required=True)
    pa.add_argument("--batch_size", type=int, default=8)
    pa.add_argument("--device", default="cuda")
    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "encode":
        from StarMoonZ1.msa import MSAEngine
        engine = MSAEngine(model_path=args.model_path, device=args.device)
        mb = engine.encode_corpus(args.corpus, output_path=args.output, batch_size=args.batch_size)
        print(f"[encode] 文档数={mb.num_docs()} 层数={mb.num_layers()} 已保存: {args.output}")

    elif args.cmd == "query":
        from StarMoonZ1.msa import MSAEngine
        from StarMoonZ1.msa.memory_bank import load_memory_bank
        engine = MSAEngine(model_path=args.model_path, device=args.device)
        engine.memory_bank = load_memory_bank(args.memory)
        if args.device.startswith("cuda"):
            engine.memory_bank.to(args.device)
        result = engine.query(
            args.prompt, top_k=args.top_k,
            use_interleave=args.interleave, max_new_tokens=args.max_new_tokens)
        print("\n===== 回答 =====")
        print(result["response"])
        print("\n===== 检索文档 =====")
        for d in result["retrieved_docs"]:
            print(f"  - {d['doc_id']} (score={d['score']:.3f})")
        print(f"\nInterleave 轮数: {result['interleave_rounds']} | 耗时: {result['latency_ms']['total']:.1f}ms")

    elif args.cmd == "add":
        from StarMoonZ1.msa import MSAEngine
        from StarMoonZ1.msa.memory_bank import load_memory_bank
        engine = MSAEngine(model_path=args.model_path, device=args.device)
        engine.memory_bank = load_memory_bank(args.memory)
        if args.device.startswith("cuda"):
            engine.memory_bank.to(args.device)
        docs, ids = [], []
        with open(args.corpus, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                docs.append(obj["text"])
                ids.append(obj.get("doc_id"))
        res = engine.add_documents(docs, doc_ids=ids, incremental=True)
        # 写回
        from StarMoonZ1.msa.memory_bank import save_memory_bank
        save_memory_bank(engine.memory_bank, args.memory)
        print(f"[add] 新增 {res['added_count']} 篇，记忆库现共 {engine.memory_bank.num_docs()} 篇，已写回 {args.memory}")


if __name__ == "__main__":
    main()
