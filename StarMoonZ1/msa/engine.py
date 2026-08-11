"""
MSA 推理引擎（in-process）与记忆库管理
======================================

MSAEngine 封装「编码语料 → 加载记忆库 → 在线查询」的完整工作流：
  - encode_corpus:   从 jsonl 批量编码文档为记忆库（Stage 1，离线）
  - load_memory_bank / save: 记忆库持久化
  - add_documents:   动态增量更新记忆库
  - query:           单轮 / Memory Interleave 多轮查询，返回答案 + 检索文档

说明：本实现为单机 in-process 版本（CPU/GPU 同进程）。文档所述的
「多 GPU Memory Parallel（KR 分片 + KV 异步 PCIe 预取）」属于生产部署优化，
可在本引擎基础上扩展，此处先提供可运行的核心实现。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch


@dataclass
class MemoryConfig:
    """记忆库配置（轻量版，对应外部 MSA 的 MemoryConfig）"""
    memory_save_path: str = "./memory_cache/"
    retrieval_top_k: int = 5
    enable_incremental_update: bool = True
    max_documents: int = 1_000_000
    max_tokens_per_doc: int = 65536
    encoding_batch_size: int = 8


class MSAEngine:
    def __init__(
        self,
        model_path: Optional[str] = None,
        config=None,
        tokenizer=None,
        device: str = "cuda",
        dtype=torch.bfloat16,
        memory_config: Optional[MemoryConfig] = None,
    ):
        from StarMoonZ1.msa.model import StarMoonZ1ForCausalLMWithMemory
        from StarMoonZ1.msa.memory_bank import load_memory_bank, MemoryLayerBank

        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config or MemoryConfig()

        if model_path is not None:
            self.model = StarMoonZ1ForCausalLMWithMemory.from_pretrained(
                model_path, torch_dtype=dtype, device_map=device, use_flash_attn=True)
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif config is not None:
            self.model = StarMoonZ1ForCausalLMWithMemory(config).to(dtype)
        else:
            raise ValueError("必须提供 model_path 或 config 之一")

        self.tokenizer = tokenizer
        self.memory_bank = None
        self._MemoryLayerBank = MemoryLayerBank
        self._load_memory_bank = load_memory_bank

    # ──────────────────────────────────────────
    # 记忆库加载 / 保存
    # ──────────────────────────────────────────
    def load_memory_bank(self, path: str, enable_lazy_load: bool = True):
        """
        加载记忆库。enable_lazy_load 时 KR/KV 先留在 CPU，查询前按需搬到 GPU
        （简化实现：整体按设备放置，后续可改为 KR 常驻 GPU / KV 异步预取）。
        """
        from StarMoonZ1.msa.memory_bank import load_memory_bank
        mb = load_memory_bank(path)
        if enable_lazy_load and self.device.startswith("cuda"):
            mb.to(self.device)  # 本机简化：整体搬到计算设备
        self.memory_bank = mb
        return mb

    def save_memory_bank(self, path: str):
        from StarMoonZ1.msa.memory_bank import save_memory_bank
        if self.memory_bank is None:
            raise RuntimeError("当前无记忆库可保存")
        return save_memory_bank(self.memory_bank, path)

    # ──────────────────────────────────────────
    # 编码
    # ──────────────────────────────────────────
    def encode_corpus(
        self,
        corpus_path: str,
        output_path: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """读取 jsonl 语料（{"doc_id","text","metadata"}），批量编码为记忆库"""
        if self.tokenizer is None:
            raise RuntimeError("encode_corpus 需要 tokenizer")
        batch_size = batch_size or self.memory_config.encoding_batch_size

        docs, metas, ids = [], [], []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                docs.append(obj["text"])
                metas.append(obj.get("metadata", {}))
                ids.append(obj.get("doc_id"))

        if not docs:
            raise ValueError(f"语料为空: {corpus_path}")

        from StarMoonZ1.msa.memory_bank import MemoryBank, MemoryLayerBank
        per_layer_acc: Dict[int, dict] = {}
        all_doc_ids, all_metas, all_lengths, all_counts = [], [], [], []

        for start in range(0, len(docs), batch_size):
            batch_docs = docs[start:start + batch_size]
            batch_ids = ids[start:start + batch_size]
            batch_meta = metas[start:start + batch_size]
            enc = self.tokenizer(
                batch_docs, return_tensors="pt", padding=True, truncation=True,
                max_length=self.memory_config.max_tokens_per_doc)
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            mb_batch = self.model.encode_documents(input_ids, attn, return_compressed=True)
            for li, lb in mb_batch.per_layer.items():
                acc = per_layer_acc.setdefault(li, {"memory_k": [], "memory_v": [], "memory_kr": [], "chunk_mask": []})
                acc["memory_k"].append(lb.memory_k)
                acc["memory_v"].append(lb.memory_v)
                acc["memory_kr"].append(lb.memory_kr)
                acc["chunk_mask"].append(lb.chunk_mask)
            all_doc_ids.extend(batch_ids)
            all_metas.extend(batch_meta)
            all_lengths.append(mb_batch.doc_lengths)
            all_counts.append(mb_batch.chunk_counts)

        per_layer = {
            li: MemoryLayerBank(
                torch.cat(acc["memory_k"], 0),
                torch.cat(acc["memory_v"], 0),
                torch.cat(acc["memory_kr"], 0),
                torch.cat(acc["chunk_mask"], 0),
            ) for li, acc in per_layer_acc.items()
        }
        mb = MemoryBank(
            per_layer=per_layer,
            doc_ids=all_doc_ids,
            doc_metadata=all_metas,
            doc_lengths=torch.cat(all_lengths, 0) if all_lengths else None,
            chunk_counts=torch.cat(all_counts, 0) if all_counts else None,
            meta={"chunk_size": self.model.config.chunk_size,
                  "router_top_k": self.model.config.router_top_k,
                  "memory_layers": self.model.config.memory_layers},
        )
        if output_path:
            from StarMoonZ1.msa.memory_bank import save_memory_bank
            save_memory_bank(mb, output_path)
        self.memory_bank = mb
        return mb

    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        incremental: bool = True,
    ) -> Dict[str, Any]:
        """动态增量添加文档到记忆库（无需重编码全库）"""
        if self.tokenizer is None:
            raise RuntimeError("add_documents 需要 tokenizer")
        if doc_ids is None:
            base = len(self.memory_bank.doc_ids) if self.memory_bank else 0
            doc_ids = [f"doc_{base + i}" for i in range(len(documents))]

        enc = self.tokenizer(
            documents, return_tensors="pt", padding=True, truncation=True,
            max_length=self.memory_config.max_tokens_per_doc)
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        mb_new = self.model.encode_documents(input_ids, attn, return_compressed=True)

        if self.memory_bank is None or not incremental:
            mb_new.doc_ids = doc_ids
            mb_new.doc_metadata = [{} for _ in documents]
            self.memory_bank = mb_new
        else:
            for li in mb_new.per_layer:
                old = self.memory_bank.per_layer[li]
                new = mb_new.per_layer[li]
                self.memory_bank.per_layer[li] = self._MemoryLayerBank(
                    torch.cat([old.memory_k, new.memory_k], 0),
                    torch.cat([old.memory_v, new.memory_v], 0),
                    torch.cat([old.memory_kr, new.memory_kr], 0),
                    torch.cat([old.chunk_mask, new.chunk_mask], 0),
                )
            self.memory_bank.doc_ids.extend(doc_ids)
            self.memory_bank.doc_metadata.extend([{} for _ in documents])
            if self.memory_bank.doc_lengths is not None:
                self.memory_bank.doc_lengths = torch.cat([self.memory_bank.doc_lengths, mb_new.doc_lengths], 0)
            if self.memory_bank.chunk_counts is not None:
                self.memory_bank.chunk_counts = torch.cat([self.memory_bank.chunk_counts, mb_new.chunk_counts], 0)

        return {"added_count": len(documents), "new_doc_ids": doc_ids}

    # ──────────────────────────────────────────
    # 查询
    # ──────────────────────────────────────────
    def query(
        self,
        prompt: str,
        top_k: Optional[int] = None,
        use_interleave: Optional[bool] = None,
        max_new_tokens: int = 2048,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        单轮 / 多轮查询。

        Returns:
            {"response", "retrieved_docs", "interleave_rounds", "latency_ms"}
        stream 参数保留接口，本实现返回非流式结果。
        """
        if self.tokenizer is None:
            raise RuntimeError("query 需要 tokenizer")
        if self.memory_bank is None:
            raise RuntimeError("未加载记忆库，请先 encode_corpus / load_memory_bank")

        top_k = top_k or self.memory_config.retrieval_top_k
        prev_k = self.model.config.router_top_k
        self.model.config.router_top_k = top_k  # 临时覆盖检索数

        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        use_interleave = (self.model.config.enable_memory_interleave
                          if use_interleave is None else use_interleave)

        t0 = time.time()
        if use_interleave:
            out_ids, trace = self.model.generate_with_interleave(
                inputs, self.memory_bank, tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens)
            response = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            last = trace[-1] if trace else None
            retrieved = last["retrieved_docs"] if last else []
            rounds = len(trace)
        else:
            out_ids = self.model.generate_with_memory(
                inputs, self.memory_bank, max_new_tokens=max_new_tokens)
            response = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            # 取检索文档：做一次带 routing 的前向（不缓存）
            rout = self.model.model(inputs, memory_bank=self.memory_bank,
                                    use_memory=True, output_routing_info=True)
            routing_infos = rout.get("routing_infos") or []
            last = next((x for x in reversed(routing_infos) if x is not None), None)
            retrieved = self.model._routing_to_docs(last, self.memory_bank, self.tokenizer)
            rounds = 1

        gen_latency = (time.time() - t0) * 1000
        self.model.config.router_top_k = prev_k

        return {
            "response": response,
            "retrieved_docs": retrieved,
            "interleave_rounds": rounds,
            "latency_ms": {"routing": 0.0, "generation": gen_latency, "total": gen_latency},
        }
