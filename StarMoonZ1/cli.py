# StarMoon-z1 CLI entry point

from __future__ import annotations
import os, sys, json, argparse, logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("starmoon-z1")


def cmd_train(args):
    from StarMoonZ1.training.trainer import TrainingArguments
    from StarMoonZ1.training.sft import SFTTrainer, SFTDataset, dynamic_padding_collate
    from StarMoonZ1.data.dataset import load_dataset
    from StarMoonZ1.model.model import StarMoonZ1ForCausalLM
    from StarMoonZ1.model.lora import LoraConfig, apply_lora
    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    data = load_dataset(args.data)
    dataset = SFTDataset(data, tokenizer, max_length=args.max_length)

    # 加载模型
    model = StarMoonZ1ForCausalLM.from_pretrained(
        args.model, use_flash_attn=not args.no_flash_attn,
    )

    # LoRA 或全量微调
    if not args.full_finetune:
        lora_cfg = LoraConfig(r=args.lora_r)
        model = apply_lora(model, lora_cfg)

    # 训练参数
    train_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_length,
        bf16=args.bf16,
    )

    trainer = SFTTrainer(model, train_args, train_dataset=dataset)
    trainer.train()
    logger.info("Training complete!")


def cmd_infer(args):
    from StarMoonZ1.inference.engine import InferenceEngine, GenerationConfig

    engine = InferenceEngine(
        model_path=args.model,
        backend=args.backend,
        use_flash_attn=not args.no_flash_attn,
    )
    engine.load_tokenizer()

    gc = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.greedy,
    )

    if args.prompt:
        output = engine.generate(args.prompt, gc)
        print(output)
    elif args.interactive:
        print("Entering interactive mode. Type 'exit' to quit.")
        while True:
            prompt = input(">>> ")
            if prompt.lower() in ("exit", "quit"):
                break
            output = engine.generate(prompt, gc)
            print(output)
    elif args.chat:
        messages = json.loads(args.chat)
        output = engine.chat(messages, gc)
        print(output)


def cmd_serve(args):
    from StarMoonZ1.inference.engine import InferenceEngine
    engine = InferenceEngine(
        model_path=args.model,
        backend=args.backend,
        use_flash_attn=not args.no_flash_attn,
    )
    engine.load_tokenizer()
    engine.create_server(host=args.host, port=args.port, webui=args.webui)


def cmd_quantize(args):
    from StarMoonZ1.model.model import StarMoonZ1ForCausalLM
    import torch

    print(f"Loading model from {args.model}...")
    model = StarMoonZ1ForCausalLM.from_pretrained(args.model)
    model = model.to(torch.float16 if args.fp16 else torch.bfloat16)

    save_path = args.output or args.model + "-quantized"
    model.save_pretrained(save_path)
    print(f"Quantized model saved to {save_path}")


def cmd_info(args):
    from StarMoonZ1.model.config import StarMoonZ1Config

    presets = {
        "1b": StarMoonZ1Config.preset_1b,
        "3b": StarMoonZ1Config.preset_3b,
        "7b": StarMoonZ1Config.preset_7b,
        "14b": StarMoonZ1Config.preset_14b,
    }

    if args.preset and args.preset in presets:
        cfg = presets[args.preset]()
        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        print(f"Estimated params: {cfg.num_params_estimate}")
    else:
        print("Available presets: 1b, 3b, 7b, 14b")
        for name, fn in presets.items():
            cfg = fn()
            print(f"  {name}: {cfg.num_params_estimate} params, "
                  f"{cfg.num_hidden_layers} layers, "
                  f"{cfg.num_attention_heads} heads, {cfg.hidden_size} hidden")


def cmd_eval(args):
    from StarMoonZ1.evaluation import Evaluator, EvalConfig, BENCHMARK_REGISTRY

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    for b in benchmarks:
        if b not in BENCHMARK_REGISTRY:
            print(f"Unknown benchmark: {b}. Available: {list(BENCHMARK_REGISTRY.keys())}")
            return

    config = EvalConfig(
        model_path=args.model,
        benchmarks=benchmarks,
        data_dir=args.data_dir,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_samples=args.max_samples,
    )

    evaluator = Evaluator(config)
    results = evaluator.run()
    evaluator.print_report(results)


def main():
    parser = argparse.ArgumentParser(description="StarMoon-z1: small model framework")
    parser.add_argument("--version", action="version", version="StarMoon-z1 0.1.0")

    sub = parser.add_subparsers(title="subcommands", dest="command")

    p_train = sub.add_parser("train", help="SFT training")
    p_train.add_argument("--model", type=str, required=True)
    p_train.add_argument("--data", type=str, required=True)
    p_train.add_argument("--output", type=str, default="./output")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--max-length", type=int, default=2048)
    p_train.add_argument("--lora-r", type=int, default=8)
    p_train.add_argument("--full-finetune", action="store_true")
    p_train.add_argument("--bf16", action="store_true", default=True)
    p_train.add_argument("--no-flash-attn", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_infer = sub.add_parser("infer", help="Inference")
    p_infer.add_argument("--model", type=str, required=True)
    p_infer.add_argument("--prompt", type=str, default=None)
    p_infer.add_argument("--chat", type=str, default=None)
    p_infer.add_argument("--interactive", action="store_true")
    p_infer.add_argument("--backend", type=str, default="auto",
                         choices=["auto", "transformers", "vllm", "llamacpp"])
    p_infer.add_argument("--max-new-tokens", type=int, default=1024)
    p_infer.add_argument("--temperature", type=float, default=0.7)
    p_infer.add_argument("--top-p", type=float, default=0.9)
    p_infer.add_argument("--greedy", action="store_true")
    p_infer.add_argument("--no-flash-attn", action="store_true")
    p_infer.set_defaults(func=cmd_infer)

    p_serve = sub.add_parser("serve", help="Start inference server (+ WebUI)")
    p_serve.add_argument("--model", type=str, required=True)
    p_serve.add_argument("--backend", type=str, default="auto")
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--webui", action="store_true", default=True,
                         help="Enable WebUI chat page at http://host:port/ (default: on)")
    p_serve.add_argument("--no-webui", dest="webui", action="store_false",
                         help="Disable WebUI (API-only mode)")
    p_serve.add_argument("--no-flash-attn", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_quant = sub.add_parser("quantize", help="Quantize model")
    p_quant.add_argument("--model", type=str, required=True)
    p_quant.add_argument("--output", type=str, default=None)
    p_quant.add_argument("--fp16", action="store_true")
    p_quant.set_defaults(func=cmd_quantize)

    p_info = sub.add_parser("info", help="Model config info")
    p_info.add_argument("--preset", type=str, choices=["1b", "3b", "7b", "14b"])
    p_info.set_defaults(func=cmd_info)

    p_eval = sub.add_parser("eval", help="Run evaluation benchmarks")
    p_eval.add_argument("--model", type=str, required=True, help="Model path")
    p_eval.add_argument("--benchmarks", type=str, default="gsm8k",
                        help="Comma-separated benchmarks: humaneval,gsm8k,mmlu,perplexity")
    p_eval.add_argument("--data-dir", type=str, default=None, help="Custom data directory")
    p_eval.add_argument("--output", type=str, default="./eval_results")
    p_eval.add_argument("--batch-size", type=int, default=8)
    p_eval.add_argument("--max-new-tokens", type=int, default=512)
    p_eval.add_argument("--temperature", type=float, default=0.0)
    p_eval.add_argument("--max-samples", type=int, default=None, help="Limit samples for debugging")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
