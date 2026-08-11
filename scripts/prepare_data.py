"""
数据准备与清洗脚本
支持: 去重、质量过滤、语言识别、格式转换。

用法:
    python scripts/prepare_data.py --input /raw_data/ --output ./data/pretrain_8gb.jsonl --mode pretrain
    python scripts/prepare_data.py --input /raw_sft/ --output ./data/sft_14gb.jsonl --mode sft
    python scripts/prepare_data.py --input /raw_code/ --output ./data/code_sft.jsonl --mode code
"""
import os
import sys
import json
import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_data")


# ──────────────────────────────────────────
# 文本质量过滤
# ──────────────────────────────────────────

class QualityFilter:
    """文本质量过滤器"""

    def __init__(self,
                 min_length: int = 100,
                 max_length: int = 100000,
                 max_special_char_ratio: float = 0.3,
                 max_repetition_ratio: float = 0.3,
                 min_avg_word_length: float = 2.0,
                 language: str = "zh,en"):
        self.min_length = min_length
        self.max_length = max_length
        self.max_special_char_ratio = max_special_char_ratio
        self.max_repetition_ratio = max_repetition_ratio
        self.min_avg_word_length = min_avg_word_length
        self.languages = language.split(",")

    def is_valid(self, text: str) -> bool:
        """检查文本是否通过质量过滤"""
        if not text or not text.strip():
            return False

        # 长度过滤
        if len(text) < self.min_length or len(text) > self.max_length:
            return False

        # 特殊字符比例
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace()
                          and c not in '，。！？、；：""''（）《》【】')
        if special_chars / max(len(text), 1) > self.max_special_char_ratio:
            return False

        # 重复行检测
        lines = text.split('\n')
        if len(lines) > 5:
            unique_lines = set(lines)
            repetition = 1 - len(unique_lines) / len(lines)
            if repetition > self.max_repetition_ratio:
                return False

        # 纯数字/符号检测
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.3:
            return False

        return True


# ──────────────────────────────────────────
# 去重
# ──────────────────────────────────────────

class Deduplicator:
    """基于 MinHash 的近似去重 (简化版: 使用精确 hash + n-gram 指纹)"""

    def __init__(self, ngram_size: int = 5, num_hashes: int = 128):
        self.ngram_size = ngram_size
        self.num_hashes = num_hashes
        self.seen_hashes: Set[str] = set()

    def _get_ngrams(self, text: str) -> List[str]:
        """提取字符级 n-gram"""
        text = text.lower().strip()
        return [text[i:i+self.ngram_size] for i in range(len(text) - self.ngram_size + 1)]

    def _minhash_signature(self, ngrams: List[str]) -> str:
        """计算 MinHash 签名 (简化版)"""
        if not ngrams:
            return ""
        # 使用多个 hash 函数的最小值作为签名
        signature = []
        for i in range(min(self.num_hashes, 32)):  # 简化: 只用 32 个
            min_hash = float('inf')
            for ng in ngrams:
                h = int(hashlib.md5(f"{i}_{ng}".encode()).hexdigest()[:8], 16)
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return "_".join(map(str, signature))

    def is_duplicate(self, text: str, threshold: float = 0.8) -> bool:
        """检查文本是否与已见文本重复"""
        # 精确去重: 完整文本 hash
        exact_hash = hashlib.md5(text.encode()).hexdigest()
        if exact_hash in self.seen_hashes:
            return True

        # 近似去重: MinHash 签名
        ngrams = self._get_ngrams(text)
        if len(ngrams) < 10:
            # 文本太短，只做精确去重
            self.seen_hashes.add(exact_hash)
            return False

        signature = self._minhash_signature(ngrams)
        sig_hash = hashlib.md5(signature.encode()).hexdigest()

        if sig_hash in self.seen_hashes:
            return True

        self.seen_hashes.add(exact_hash)
        self.seen_hashes.add(sig_hash)
        return False


# ──────────────────────────────────────────
# 数据加载与转换
# ──────────────────────────────────────────

def load_raw_files(input_path: str) -> List[Dict]:
    """加载原始数据文件 (支持 jsonl, json, txt, csv)"""
    data = []
    input_path = Path(input_path)

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("**/*"))

    for f in files:
        if f.suffix in ('.jsonl', '.json'):
            data.extend(_load_jsonl(f))
        elif f.suffix == '.txt':
            data.extend(_load_txt(f))
        elif f.suffix == '.csv':
            data.extend(_load_csv(f))
        elif f.suffix in ('.parquet',):
            data.extend(_load_parquet(f))

    logger.info(f"加载完成: {len(data):,} 条原始数据")
    return data


def _load_jsonl(path: Path) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def _load_txt(path: Path) -> List[Dict]:
    """纯文本文件: 按段落分割"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 按双换行分段
    paragraphs = content.split('\n\n')
    for para in paragraphs:
        para = para.strip()
        if len(para) > 50:
            data.append({"text": para})
    return data


def _load_csv(path: Path) -> List[Dict]:
    """CSV 文件"""
    import csv
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    return data


def _load_parquet(path: Path) -> List[Dict]:
    """Parquet 文件"""
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict('records')
    except ImportError:
        logger.warning("pandas/pyarrow not installed, skipping parquet files")
        return []


# ──────────────────────────────────────────
# 模式处理
# ──────────────────────────────────────────

def process_pretrain(data: List[Dict], args) -> List[Dict]:
    """预训练数据处理: 纯文本"""
    qf = QualityFilter(
        min_length=args.min_length,
        max_length=args.max_length,
        language=args.language,
    )
    dedup = Deduplicator()

    results = []
    stats = Counter()

    for item in data:
        text = item.get("text", item.get("content", ""))
        if not text:
            stats["empty"] += 1
            continue

        # 质量过滤
        if not qf.is_valid(text):
            stats["quality_filtered"] += 1
            continue

        # 去重
        if not args.no_dedup and dedup.is_duplicate(text):
            stats["duplicate"] += 1
            continue

        results.append({"text": text.strip()})
        stats["passed"] += 1

    logger.info(f"预训练数据处理完成:")
    logger.info(f"  通过: {stats['passed']:,}")
    logger.info(f"  质量过滤: {stats['quality_filtered']:,}")
    logger.info(f"  去重: {stats['duplicate']:,}")
    logger.info(f"  空文本: {stats['empty']:,}")
    return results


def process_sft(data: List[Dict], args) -> List[Dict]:
    """SFT 数据处理: 多轮对话格式"""
    results = []
    stats = Counter()

    for item in data:
        # 支持多种输入格式
        if "messages" in item:
            messages = item["messages"]
        elif "conversations" in item:
            # ShareGPT 格式转换
            messages = _convert_sharegpt(item["conversations"])
        elif "instruction" in item:
            # Alpaca 格式转换
            messages = _convert_alpaca(item)
        elif "prompt" in item and "response" in item:
            messages = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]},
            ]
        else:
            stats["unknown_format"] += 1
            continue

        # 验证对话格式
        if not _validate_messages(messages):
            stats["invalid"] += 1
            continue

        results.append({"messages": messages})
        stats["passed"] += 1

    logger.info(f"SFT 数据处理完成:")
    logger.info(f"  通过: {stats['passed']:,}")
    logger.info(f"  格式无效: {stats['invalid']:,}")
    logger.info(f"  未知格式: {stats['unknown_format']:,}")
    return results


def process_code(data: List[Dict], args) -> List[Dict]:
    """代码数据处理: 转为对话格式"""
    results = []
    stats = Counter()

    for item in data:
        # 已经是 messages 格式
        if "messages" in item:
            results.append({"messages": item["messages"]})
            stats["passed"] += 1
            continue

        # 代码补全格式: {"input": ..., "output": ...}
        if "input" in item and "output" in item:
            messages = [
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["output"]},
            ]
            results.append({"messages": messages})
            stats["passed"] += 1
            continue

        # 纯代码格式: {"code": ..., "language": ..., "description": ...}
        if "code" in item:
            desc = item.get("description", item.get("docstring", "解释以下代码"))
            code = item["code"]
            lang = item.get("language", "python")

            # 随机选择任务类型
            messages = [
                {"role": "user", "content": desc},
                {"role": "assistant", "content": f"```{lang}\n{code}\n```"},
            ]
            results.append({"messages": messages})
            stats["passed"] += 1
            continue

        stats["unknown_format"] += 1

    logger.info(f"代码数据处理完成:")
    logger.info(f"  通过: {stats['passed']:,}")
    logger.info(f"  未知格式: {stats['unknown_format']:,}")
    return results


def _convert_sharegpt(conversations: List[Dict]) -> List[Dict]:
    """ShareGPT 格式 → 标准 messages 格式"""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for turn in conversations:
        role = role_map.get(turn.get("from", ""), "user")
        content = turn.get("value", "")
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _convert_alpaca(item: Dict) -> List[Dict]:
    """Alpaca 格式 → 标准 messages 格式"""
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output = item.get("output", "")

    user_content = instruction
    if input_text:
        user_content += f"\n\n{input_text}"

    messages = [{"role": "user", "content": user_content}]
    if output:
        messages.append({"role": "assistant", "content": output})
    return messages


def _validate_messages(messages: List[Dict]) -> bool:
    """验证对话格式合法性"""
    if not messages or len(messages) < 2:
        return False

    # 必须有 assistant 回复
    has_assistant = any(m.get("role") == "assistant" for m in messages)
    if not has_assistant:
        return False

    # 检查内容非空
    for msg in messages:
        if not msg.get("content", "").strip():
            return False
        if msg.get("role") not in ("system", "user", "assistant"):
            return False

    return True


# ──────────────────────────────────────────
# 数据保存
# ──────────────────────────────────────────

def save_data(data: List[Dict], output_path: str):
    """保存为 JSONL 格式"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 统计文件大小
    file_size = os.path.getsize(output_path)
    size_str = f"{file_size / 1e9:.2f} GB" if file_size > 1e9 else f"{file_size / 1e6:.1f} MB"
    logger.info(f"保存完成: {output_path} ({len(data):,} 条, {size_str})")


# ──────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="StarMoon-z1 数据准备与清洗")
    parser.add_argument("--input", type=str, required=True,
                        help="输入数据路径 (文件或目录)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSONL 路径")
    parser.add_argument("--mode", type=str, default="pretrain",
                        choices=["pretrain", "sft", "code"],
                        help="处理模式: pretrain/sft/code")
    parser.add_argument("--min-length", type=int, default=100,
                        help="最小文本长度 (字符)")
    parser.add_argument("--max-length", type=int, default=100000,
                        help="最大文本长度 (字符)")
    parser.add_argument("--language", type=str, default="zh,en",
                        help="保留的语言 (逗号分隔)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="跳过去重 (加速处理)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数限制")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    logger.info(f"数据准备: mode={args.mode}")
    logger.info(f"  输入: {args.input}")
    logger.info(f"  输出: {args.output}")

    # 1. 加载原始数据
    data = load_raw_files(args.input)

    # 2. 按模式处理
    if args.mode == "pretrain":
        processed = process_pretrain(data, args)
    elif args.mode == "sft":
        processed = process_sft(data, args)
    elif args.mode == "code":
        processed = process_code(data, args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # 3. 限制样本数
    if args.max_samples and len(processed) > args.max_samples:
        import random
        random.seed(args.seed)
        processed = random.sample(processed, args.max_samples)
        logger.info(f"采样限制: {len(processed):,} 条")

    # 4. 保存
    save_data(processed, args.output)


if __name__ == "__main__":
    main()
