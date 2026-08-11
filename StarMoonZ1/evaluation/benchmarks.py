"""
内置 Benchmark 定义
支持: HumanEval (代码), GSM8K (数学), MMLU (知识), Perplexity (语言建模)。
每个 Benchmark 提供: 数据加载、prompt 构建、答案提取、评分。
"""
from __future__ import annotations
import json, os, re, logging, threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from StarMoonZ1.evaluation.metrics import (
    pass_at_k, exact_match, accuracy, perplexity,
    code_extract_answer, math_extract_answer,
)

logger = logging.getLogger("StarMoonZ1.Eval")


@dataclass
class EvalSample:
    """单条评测样本"""
    id: str
    prompt: str
    reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """单条评测结果"""
    id: str
    prompt: str
    generation: str
    reference: str
    score: float
    correct: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class Benchmark(ABC):
    """Benchmark 基类"""
    name: str = "base"
    metric_name: str = "score"

    @abstractmethod
    def load_data(self, data_path: Optional[str] = None) -> List[EvalSample]:
        """加载评测数据"""
        ...

    @abstractmethod
    def build_prompt(self, sample: EvalSample) -> str:
        """构建评测 prompt"""
        ...

    @abstractmethod
    def score(self, sample: EvalSample, generation: str) -> float:
        """对单条生成结果评分"""
        ...

    def aggregate(self, results: List[EvalResult]) -> Dict[str, float]:
        """聚合所有结果为最终指标"""
        if not results:
            return {self.metric_name: 0.0}
        scores = [r.score for r in results]
        return {
            self.metric_name: sum(scores) / len(scores),
            "num_samples": len(results),
            "num_correct": sum(1 for r in results if r.correct),
        }


# ──────────────────────────────────────────
# HumanEval (代码生成)
# ──────────────────────────────────────────

class HumanEvalBenchmark(Benchmark):
    """
    HumanEval 代码生成评测。
    数据格式 (JSONL): {"task_id": "...", "prompt": "def ...", "entry_point": "...", "test": "...", "canonical_solution": "..."}
    """
    name = "humaneval"
    metric_name = "pass@1"

    def __init__(self, k: int = 1, num_samples: int = 1, timeout: float = 5.0):
        self.k = k
        self.num_samples = num_samples
        self.timeout = timeout

    def load_data(self, data_path: Optional[str] = None) -> List[EvalSample]:
        path = data_path or self._default_path()
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                samples.append(EvalSample(
                    id=item["task_id"],
                    prompt=item["prompt"],
                    reference=item.get("canonical_solution", ""),
                    metadata={"entry_point": item.get("entry_point", ""),
                              "test": item.get("test", "")},
                ))
        return samples

    def build_prompt(self, sample: EvalSample) -> str:
        return sample.prompt  # HumanEval prompt 已经是函数签名

    def score(self, sample: EvalSample, generation: str) -> float:
        """执行代码并运行测试用例 (沙箱隔离)"""
        code = code_extract_answer(generation, sample.metadata.get("entry_point", ""))
        full_code = sample.prompt + code
        test_code = sample.metadata.get("test", "")
        entry_point = sample.metadata.get("entry_point", "")
        check_code = f"{full_code}\n{test_code}\ncheck({entry_point})"
        return _safe_exec_code(check_code, timeout=self.timeout)

    def aggregate(self, results: List[EvalResult]) -> Dict[str, float]:
        if not results:
            return {"pass@1": 0.0}
        correct = sum(1 for r in results if r.correct)
        n = len(results)
        return {
            "pass@1": pass_at_k(n, correct, 1),
            "pass@10": pass_at_k(n, correct, min(10, n)),
            "accuracy": correct / n,
            "num_samples": n,
            "num_correct": correct,
        }

    def _default_path(self):
        return os.path.join(os.path.dirname(__file__), "data", "humaneval.jsonl")


# ──────────────────────────────────────────
# GSM8K (数学推理)
# ──────────────────────────────────────────

class GSM8KBenchmark(Benchmark):
    """
    GSM8K 数学推理评测。
    数据格式 (JSONL): {"question": "...", "answer": "... #### 42"}
    """
    name = "gsm8k"
    metric_name = "accuracy"

    FEW_SHOT_PREFIX = (
        "Question: There are 15 trees in the grove. Grove workers will plant trees today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
        "Answer: We started with 15 trees. After planting, there are 21 trees. "
        "So the workers planted 21 - 15 = 6 trees. #### 6\n\n"
    )

    def load_data(self, data_path: Optional[str] = None) -> List[EvalSample]:
        path = data_path or self._default_path()
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line.strip())
                answer_text = item["answer"]
                # 提取 #### 后的数字
                final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else ""
                samples.append(EvalSample(
                    id=f"gsm8k_{i}",
                    prompt=item["question"],
                    reference=final_answer.replace(",", ""),
                    metadata={"full_answer": answer_text},
                ))
        return samples

    def build_prompt(self, sample: EvalSample) -> str:
        return f"{self.FEW_SHOT_PREFIX}Question: {sample.prompt}\nAnswer: Let's think step by step.\n"

    def score(self, sample: EvalSample, generation: str) -> float:
        predicted = math_extract_answer(generation)
        reference = sample.reference.replace(",", "").strip()
        return 1.0 if predicted == reference else 0.0

    def _default_path(self):
        return os.path.join(os.path.dirname(__file__), "data", "gsm8k.jsonl")


# ──────────────────────────────────────────
# MMLU (多学科知识)
# ──────────────────────────────────────────

class MMLUBenchmark(Benchmark):
    """
    MMLU 多学科选择题评测。
    数据格式 (JSONL): {"question": "...", "A": "...", "B": "...", "C": "...", "D": "...", "answer": "A", "subject": "..."}
    """
    name = "mmlu"
    metric_name = "accuracy"
    CHOICES = ["A", "B", "C", "D"]

    def load_data(self, data_path: Optional[str] = None) -> List[EvalSample]:
        path = data_path or self._default_path()
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line.strip())
                samples.append(EvalSample(
                    id=f"mmlu_{i}",
                    prompt=item["question"],
                    reference=item["answer"],
                    metadata={
                        "choices": [item.get("A", ""), item.get("B", ""),
                                    item.get("C", ""), item.get("D", "")],
                        "subject": item.get("subject", "general"),
                    },
                ))
        return samples

    def build_prompt(self, sample: EvalSample) -> str:
        choices = sample.metadata["choices"]
        subject = sample.metadata.get("subject", "general").replace("_", " ")
        prompt = f"The following is a multiple choice question about {subject}.\n\n"
        prompt += f"{sample.prompt}\n"
        for i, (label, text) in enumerate(zip(self.CHOICES, choices)):
            prompt += f"{label}. {text}\n"
        prompt += "\nAnswer:"
        return prompt

    def score(self, sample: EvalSample, generation: str) -> float:
        # 提取模型选择的字母
        predicted = self._extract_choice(generation)
        return 1.0 if predicted == sample.reference.strip().upper() else 0.0

    def aggregate(self, results: List[EvalResult]) -> Dict[str, float]:
        base = super().aggregate(results)
        # 按 subject 分组统计 (subject 存于 EvalResult.metadata)
        subject_scores: Dict[str, List[float]] = {}
        for r in results:
            subject = r.metadata.get("subject", "general")
            subject_scores.setdefault(subject, []).append(r.score)
        if subject_scores:
            base["per_subject"] = {
                subj: sum(scores) / len(scores)
                for subj, scores in subject_scores.items()
            }
        return base

    def _extract_choice(self, generation: str) -> str:
        gen = generation.strip().upper()
        # 直接以字母开头
        if gen and gen[0] in self.CHOICES:
            return gen[0]
        # 查找 "A." / "A)" / "(A)" 模式
        match = re.search(r'[(\s]?([A-D])[).]', gen)
        if match:
            return match.group(1)
        # 查找 "the answer is X"
        match = re.search(r'(?:ANSWER|answer)\s*(?:IS|is|:)\s*([A-D])', generation)
        if match:
            return match.group(1).upper()
        return ""

    def _default_path(self):
        return os.path.join(os.path.dirname(__file__), "data", "mmlu.jsonl")


# ──────────────────────────────────────────
# Perplexity (语言建模)
# ──────────────────────────────────────────

class PerplexityBenchmark(Benchmark):
    """
    困惑度评测: 在给定文本上计算 PPL。
    数据格式 (JSONL): {"text": "..."}
    """
    name = "perplexity"
    metric_name = "ppl"

    def load_data(self, data_path: Optional[str] = None) -> List[EvalSample]:
        path = data_path or self._default_path()
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line.strip())
                samples.append(EvalSample(
                    id=f"ppl_{i}",
                    prompt=item["text"],
                    reference="",
                ))
        return samples

    def build_prompt(self, sample: EvalSample) -> str:
        return sample.prompt  # PPL 不需要生成，直接计算 loss

    def score(self, sample: EvalSample, generation: str) -> float:
        # PPL 的 score 由 evaluator 特殊处理 (通过 loss 计算)
        return 0.0

    def aggregate(self, results: List[EvalResult]) -> Dict[str, float]:
        # 由 evaluator 覆盖: 使用 total_loss / total_tokens
        return {"ppl": 0.0, "num_samples": len(results)}

    def _default_path(self):
        return os.path.join(os.path.dirname(__file__), "data", "perplexity.jsonl")


# ──────────────────────────────────────────
# 注册表
# ──────────────────────────────────────────

BENCHMARK_REGISTRY: Dict[str, type] = {
    "humaneval": HumanEvalBenchmark,
    "gsm8k": GSM8KBenchmark,
    "mmlu": MMLUBenchmark,
    "perplexity": PerplexityBenchmark,
}


def get_benchmark(name: str, **kwargs) -> Benchmark:
    """按名称获取 benchmark 实例"""
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_REGISTRY.keys())}")
    return BENCHMARK_REGISTRY[name](**kwargs)


# ──────────────────────────────────────────
# 安全代码执行 (沙箱)
# ──────────────────────────────────────────

def _safe_exec_code(code: str, timeout: float = 5.0) -> float:
    """
    在受限子进程中执行代码，防止恶意代码影响主进程。
    
    安全措施:
    - 子进程隔离: 代码在独立进程中运行，崩溃不影响主进程
    - 超时强制终止: 防止死循环
    - 限制 builtins: 禁止文件/网络/系统操作
    - 回退机制: 子进程不可用时仍通过 builtins 限制提供基本保护
    """
    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
    from concurrent.futures.process import BrokenProcessPool
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_exec_worker_fn, code)
            result = future.result(timeout=timeout)
            return result
    except FuturesTimeout:
        return 0.0
    except (OSError, RuntimeError, BrokenProcessPool):
        # Windows spawn 失败时回退: 仍施加受限 builtins，并放入守护线程加超时，
        # 以恢复进程池缺失时丢失的超时保护，避免主进程被恶意/死循环代码挂起。
        return _exec_with_timeout(code, timeout)
    except Exception:
        return 0.0


def _exec_worker_fn(code: str) -> float:
    """子进程工作函数: 在受限环境中执行代码"""
    import builtins as _builtins
    safe_builtins = {
        k: getattr(_builtins, k) for k in (
            "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod",
            "enumerate", "filter", "float", "format", "frozenset", "hash",
            "hex", "id", "int", "isinstance", "issubclass", "iter", "len",
            "list", "map", "max", "min", "next", "oct", "ord", "pow",
            "print", "range", "repr", "reversed", "round", "set", "slice",
            "sorted", "str", "sum", "tuple", "type", "zip",
        ) if hasattr(_builtins, k)
    }
    safe_builtins["__import__"] = _restricted_import
    exec_globals = {"__builtins__": safe_builtins}
    try:
        exec(code, exec_globals)
        return 1.0
    except Exception:
        return 0.0


def _exec_with_timeout(code: str, timeout: float) -> float:
    """受限环境下的带超时执行 (Windows 进程池不可用时的安全回退)。

    进程隔离已丢失，但至少通过守护线程 + join(timeout) 恢复超时保护，
    并复用 _exec_worker_fn 的受限 builtins，避免主线程被挂起。
    """
    result: Dict[str, float] = {"val": 0.0}

    def _run():
        try:
            result["val"] = _exec_worker_fn(code)
        except Exception:
            result["val"] = 0.0

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        # 超时: 守护线程将在解释器退出时终止，这里直接判 0
        return 0.0
    return result["val"]


def _restricted_import(name, *args, **kwargs):
    """仅允许导入安全的标准库模块"""
    allowed_modules = {
        "math", "cmath", "decimal", "fractions", "random", "statistics",
        "itertools", "functools", "operator", "collections", "heapq",
        "bisect", "copy", "pprint", "string", "re", "textwrap", "unicodedata",
        "struct", "codecs", "datetime", "calendar", "abc", "typing",
        "dataclasses", "enum", "numbers", "array",
    }
    top_level = name.split(".")[0]
    if top_level not in allowed_modules:
        raise ImportError(f"Import of '{name}' is not allowed in evaluation sandbox")
    import importlib
    return importlib.import_module(name)
