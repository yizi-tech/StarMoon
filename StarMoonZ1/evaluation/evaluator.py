"""
统一评测引擎
负责: 模型加载、批量生成、评分调度、结果聚合、JSON 报告输出。
"""
from __future__ import annotations
import os, json, time, logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

import torch
import torch.nn.functional as F

from StarMoonZ1.evaluation.benchmarks import (
    Benchmark, EvalSample, EvalResult, get_benchmark, BENCHMARK_REGISTRY,
    PerplexityBenchmark,
)
from StarMoonZ1.evaluation.metrics import perplexity as calc_perplexity

logger = logging.getLogger("StarMoonZ1.Eval")


@dataclass
class EvalConfig:
    """评测配置"""
    model_path: str = ""
    benchmarks: List[str] = field(default_factory=lambda: ["gsm8k"])
    data_dir: Optional[str] = None          # 自定义数据目录
    output_dir: str = "./eval_results"
    batch_size: int = 8                     # 预留: 生成式评测目前逐条处理, 未启用 batch
    max_new_tokens: int = 512
    temperature: float = 0.0                # 评测默认 greedy
    top_p: float = 1.0
    max_samples: Optional[int] = None       # 限制评测样本数 (调试用)
    backend: str = "transformers"           # 推理后端: transformers / vllm / llamacpp
    device: str = "auto"                    # auto 表示跟随模型权重所在设备
    # 各 benchmark 构造参数, 如 {"humaneval": {"k": 10, "num_samples": 10, "timeout": 10.0}}
    benchmark_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class Evaluator:
    """
    统一评测引擎。
    
    用法:
        evaluator = Evaluator(config)
        results = evaluator.run()
        evaluator.print_report(results)
    """
    def __init__(self, config: EvalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None

    def _bench_kwargs(self, name: str) -> Dict[str, Any]:
        """从 config.benchmark_kwargs 取指定 benchmark 的构造参数"""
        return dict(self.config.benchmark_kwargs.get(name, {}))

    def run(self) -> Dict[str, Dict[str, float]]:
        """执行全部 benchmark 评测"""
        self._load_model()
        all_results = {}

        for bench_name in self.config.benchmarks:
            logger.info(f"{'='*60}")
            logger.info(f"Running benchmark: {bench_name}")
            logger.info(f"{'='*60}")

            benchmark = get_benchmark(bench_name, **self._bench_kwargs(bench_name))
            data_path = self._resolve_data_path(bench_name)
            samples = benchmark.load_data(data_path)

            if self.config.max_samples:
                samples = samples[:self.config.max_samples]

            logger.info(f"  Loaded {len(samples)} samples")

            if isinstance(benchmark, PerplexityBenchmark):
                metrics = self._eval_perplexity(benchmark, samples)
            else:
                metrics = self._eval_generation(benchmark, samples)

            all_results[bench_name] = metrics
            logger.info(f"  Results: {metrics}")

        # 保存报告
        self._save_report(all_results)
        return all_results

    # ──────────────────────────────────────────
    # 模型加载
    # ──────────────────────────────────────────

    def _load_model(self):
        from StarMoonZ1.model.model import StarMoonZ1ForCausalLM
        from transformers import AutoTokenizer

        logger.info(f"Loading model from {self.config.model_path} "
                    f"(backend={self.config.backend})...")
        self.model = StarMoonZ1ForCausalLM.from_pretrained(
            self.config.model_path, use_flash_attn=True)
        self.model.eval()

        # 显式指定设备 (auto = 跟随模型权重所在设备)
        if self.config.device and self.config.device != "auto":
            self.model = self.model.to(self.config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = next(self.model.parameters()).device
        logger.info(f"  Model loaded on {self.device}")

    # ──────────────────────────────────────────
    # 生成式评测 (HumanEval, GSM8K, MMLU)
    # ──────────────────────────────────────────

    @torch.no_grad()
    def _eval_generation(self, benchmark: Benchmark, samples: List[EvalSample]) -> Dict[str, float]:
        results: List[EvalResult] = []
        start_time = time.time()

        for i, sample in enumerate(samples):
            prompt = benchmark.build_prompt(sample)
            generation = self._generate(prompt)
            score = benchmark.score(sample, generation)

            results.append(EvalResult(
                id=sample.id,
                prompt=prompt,
                generation=generation,
                reference=sample.reference,
                score=score,
                correct=score >= 0.5,
                metadata=dict(sample.metadata),
            ))

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  Progress: {i+1}/{len(samples)} | "
                            f"Acc so far: {sum(r.score for r in results)/len(results):.3f} | "
                            f"{elapsed:.1f}s")

        metrics = benchmark.aggregate(results)
        metrics["time_seconds"] = time.time() - start_time
        return metrics

    def _generate(self, prompt: str) -> str:
        """单条生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=self.config.max_new_tokens,
            temperature=max(self.config.temperature, 0.01),
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = output_ids[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    # ──────────────────────────────────────────
    # Perplexity 评测 (特殊: 不生成, 计算 loss)
    # ──────────────────────────────────────────

    @torch.no_grad()
    def _eval_perplexity(self, benchmark: PerplexityBenchmark,
                         samples: List[EvalSample]) -> Dict[str, float]:
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()

        for i, sample in enumerate(samples):
            enc = self.tokenizer(sample.prompt, return_tensors="pt",
                                 truncation=True, max_length=2048).to(self.device)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")
            labels = input_ids.clone()

            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            # 仅统计非 pad token (用 attention_mask 排除 padding)
            if attention_mask is not None:
                # labels 右移 1 位后与 attention_mask 对齐, 排除 padding 的预测位
                valid_mask = attention_mask[:, 1:].to(input_ids.dtype)
                num_tokens = valid_mask.sum().item()
            else:
                num_tokens = input_ids.shape[1] - 1
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if (i + 1) % 100 == 0:
                current_ppl = calc_perplexity(total_loss, total_tokens)
                logger.info(f"  PPL progress: {i+1}/{len(samples)} | PPL={current_ppl:.2f}")

        ppl = calc_perplexity(total_loss, total_tokens)
        return {
            "ppl": ppl,
            "num_samples": len(samples),
            "total_tokens": total_tokens,
            "time_seconds": time.time() - start_time,
        }

    # ──────────────────────────────────────────
    # 报告
    # ──────────────────────────────────────────

    def _resolve_data_path(self, bench_name: str) -> Optional[str]:
        if self.config.data_dir:
            path = os.path.join(self.config.data_dir, f"{bench_name}.jsonl")
            if os.path.exists(path):
                return path
        return None  # 使用 benchmark 默认路径

    def _save_report(self, results: Dict[str, Dict[str, float]]):
        os.makedirs(self.config.output_dir, exist_ok=True)
        report = {
            "model": self.config.model_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": asdict(self.config),
            "results": results,
        }
        path = os.path.join(self.config.output_dir, "eval_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {path}")

    def print_report(self, results: Dict[str, Dict[str, float]]):
        """打印格式化评测报告"""
        print("\n" + "=" * 60)
        print(f"  EVALUATION REPORT")
        print(f"  Model: {self.config.model_path}")
        print("=" * 60)
        for bench_name, metrics in results.items():
            print(f"\n  [{bench_name.upper()}]")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        print("\n" + "=" * 60)
