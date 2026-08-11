"""
评测指标模块
提供: pass@k, exact_match, accuracy, perplexity, F1 等标准指标。
"""
from __future__ import annotations
import math
import numpy as np
from typing import List, Optional
from collections import Counter


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    计算 pass@k 指标 (无偏估计)。
    
    Args:
        n: 总生成样本数
        c: 正确样本数
        k: k 值
    Returns:
        pass@k 概率
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def exact_match(prediction: str, reference: str, normalize: bool = True) -> float:
    """精确匹配 (支持大小写/空白归一化)"""
    if normalize:
        prediction = _normalize(prediction)
        reference = _normalize(reference)
    return 1.0 if prediction == reference else 0.0


def accuracy(predictions: List[str], references: List[str], normalize: bool = True) -> float:
    """批量精确匹配准确率"""
    if not predictions:
        return 0.0
    scores = [exact_match(p, r, normalize) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


def perplexity(total_loss: float, total_tokens: int) -> float:
    """
    计算困惑度 PPL = exp(avg_cross_entropy)。
    
    Args:
        total_loss: 总交叉熵损失 (sum)
        total_tokens: 有效 token 数
    """
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def f1_score(prediction: str, reference: str) -> float:
    """Token 级别 F1 (适用于生成式问答)"""
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def code_extract_answer(generation: str, entry_point: str = "") -> str:
    """从模型生成中提取代码 (用于 HumanEval 等代码评测)"""
    # 尝试提取 ```code``` 块
    if "```" in generation:
        parts = generation.split("```")
        for part in parts[1::2]:  # 奇数位是代码块内容
            lines = part.strip().split("\n")
            # 跳过语言标识行
            if lines and lines[0].strip() in ("python", "py", ""):
                lines = lines[1:]
            return "\n".join(lines)
    # 无代码块: 截取到第一个空行或 def/class 之后
    lines = generation.split("\n")
    result = []
    for line in lines:
        result.append(line)
        # 遇到顶层 return 或空行结束
        if line.strip().startswith("return ") and not line.startswith(" "):
            break
    return "\n".join(result)


def math_extract_answer(generation: str) -> str:
    """从模型生成中提取数学答案 (GSM8K 格式: #### 数字)"""
    # 查找 #### 标记
    if "####" in generation:
        answer = generation.split("####")[-1].strip()
        return _extract_number(answer)
    # 查找 "the answer is" 模式
    lower = generation.lower()
    for pattern in ["the answer is", "answer:", "final answer"]:
        if pattern in lower:
            idx = lower.rfind(pattern) + len(pattern)
            return _extract_number(generation[idx:].strip())
    # 取最后一行的数字
    lines = generation.strip().split("\n")
    if lines:
        return _extract_number(lines[-1])
    return ""


def _extract_number(text: str) -> str:
    """从文本中提取数字 (支持负数、小数、逗号分隔)"""
    text = text.replace(",", "").strip()
    result = []
    started = False
    for ch in text:
        if ch in "0123456789.-":
            result.append(ch)
            started = True
        elif started:
            break
    return "".join(result)


def _normalize(text: str) -> str:
    """文本归一化: 小写 + 去首尾空白 + 压缩空格"""
    return " ".join(text.lower().strip().split())
