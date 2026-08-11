"""
数据处理模块 - 数据集加载与预处理
支持: JSONL 加载、流式数据集、领域混合、对话模板。
"""
from __future__ import annotations
import json, os, random, glob
from typing import Optional, List, Dict, Any, Iterator
import torch
from torch.utils.data import Dataset, IterableDataset


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer=None, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.tokenizer:
            enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 padding="max_length", return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": enc["input_ids"].squeeze(0).clone()}
        return {"text": text}


def load_dataset(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_chat_template(messages: List[Dict[str, str]], tokenizer=None) -> str:
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            text += f"<|system|>\n{content}\n"
        elif role == "user":
            text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}\n"
    return text


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 2048) -> Dataset:
    class TokenizedDataset(Dataset):
        def __init__(self, data, tok, ml):
            self.data = data
            self.tok = tok
            self.ml = ml
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            text = item.get("text", item.get("content", str(item)))
            enc = self.tok(text, truncation=True, max_length=self.ml, padding="max_length", return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": enc["input_ids"].squeeze(0).clone()}
    return TokenizedDataset(dataset, tokenizer, max_length)


# ──────────────────────────────────────────
# 流式数据集 (大规模预训练)
# ──────────────────────────────────────────

class StreamingTextDataset(IterableDataset):
    """
    流式文本数据集: 逐行读取 JSONL，无需全量加载到内存。
    适用于 TB 级预训练数据。
    """
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048,
                 text_field: str = "text", shuffle_buffer: int = 10000):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.shuffle_buffer = shuffle_buffer
        # 支持目录 (多文件) 或单文件
        if os.path.isdir(data_path):
            self.files = sorted(glob.glob(os.path.join(data_path, "*.jsonl")) +
                                glob.glob(os.path.join(data_path, "*.json")))
        else:
            self.files = [data_path]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buf = []
        for fpath in self.files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    text = item.get(self.text_field, item.get("content", ""))
                    if not text:
                        continue
                    buf.append(text)
                    if len(buf) >= self.shuffle_buffer:
                        random.shuffle(buf)
                        for t in buf:
                            yield self._tokenize(t)
                        buf = []
        # 剩余
        if buf:
            random.shuffle(buf)
            for t in buf:
                yield self._tokenize(t)

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0).clone()}


# ──────────────────────────────────────────
# 领域混合数据集 (代码/数学/通用)
# ──────────────────────────────────────────

class DomainMixedDataset(Dataset):
    """
    领域混合数据集: 按权重混合多个领域数据。
    
    使用确定性采样 (基于 idx 的 seed)，保证 DataLoader shuffle 可复现。
    
    典型用法 (训练最强 1B 代码/推理模型):
        dataset = DomainMixedDataset(
            sources={
                "code": (code_data, 0.35),      # 35% 代码
                "math": (math_data, 0.25),      # 25% 数学推理
                "general": (general_data, 0.30), # 30% 通用文本
                "reasoning": (reason_data, 0.10), # 10% 逻辑推理
            },
            tokenizer=tokenizer,
            max_length=4096,
        )
    """
    def __init__(self, sources: Dict[str, tuple], tokenizer, max_length: int = 2048,
                 total_samples: Optional[int] = None, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.domains = []
        self.weights = []
        self.cumulative = []

        total = 0
        for name, (data, weight) in sources.items():
            self.domains.append(data)
            self.weights.append(weight)
            total += len(data)

        # 归一化权重
        w_sum = sum(self.weights)
        self.weights = [w / w_sum for w in self.weights]

        # 累计分布 (用于按权重采样)
        cum = 0.0
        for w in self.weights:
            cum += w
            self.cumulative.append(cum)

        self.total_samples = total_samples or total

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 确定性采样: 基于 idx + seed 生成局部随机数，保证可复现
        rng = random.Random(self.seed + idx)
        r = rng.random()
        domain_idx = 0
        for i, c in enumerate(self.cumulative):
            if r <= c:
                domain_idx = i
                break
        domain_data = self.domains[domain_idx]
        item_idx = rng.randint(0, len(domain_data) - 1)
        item = domain_data[item_idx]
        text = item.get("text", item.get("content", str(item))) if isinstance(item, dict) else str(item)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0).clone()}
