#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_tokenizer.py
==================
StarMoon-z1 自有分词器训练 (Unigram ~150k)。

依据: docs/tokenizer-design.md
流程: 采样语料(corpus.jsonl) → Unigram 训练 → 注入特殊 token(预留 11–31)
      → 导出 HF 格式 → 自动同步 StarMoonZ1/model/config.py (词表大小 + 特殊 token id)

特性:
- 基于 `tokenizers` 训练 Unigram，再用 `transformers.PreTrainedTokenizerFast` 包装导出，
  产出 tokenizer.json / tokenizer_config.json(含 chat_template) / special_tokens_map.json，
  可被 `AutoTokenizer.from_pretrained` 即插即用。
- 特殊 token 按 docs/tokenizer-design.md §4 固定 id:
    <pad>=0 <unk>=1 <bos>=2 <eos>=3 <|system|>=4 <|user|>=5 <|assistant|>=6
    <image>=7 <video>=8 <audio>=9 <think>=10，并预留 <reserved_11>..<reserved_31>。
- 训练后自动改写 config.py: 默认 + 4 个 preset 的 vocab_size、from_hf 回读默认值，
  以及 bos/eos/image_token_id（保留 .bak 备份）。

用法 (AutoDL, 接 download_tokenizer_corpus.py 的输出):
    pip install tokenizers transformers
    python scripts/train_tokenizer.py \
        --corpus /root/data/tok_corpus/raw/corpus.jsonl \
        --output-dir /root/data/starmoon-tokenizer \
        --vocab-size 150000

仅训练分词器；模型训练请用 scripts/train_cpt.py，并把 BASE_MODEL 指向本分词器目录。
"""
import os
import re
import sys
import json
import shutil
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("TrainTok")

# ──────────────────────────────────────────────
# 特殊 token（顺序即 id 分配顺序，勿改）
# ──────────────────────────────────────────────
SPECIAL_ORDER = [
    "<pad>",          # 0
    "<unk>",          # 1
    "<bos>",          # 2
    "<eos>",          # 3
    "<|system|>",     # 4
    "<|user|>",       # 5
    "<|assistant|>",  # 6
    "<image>",        # 7  (y1 / Ω 复用)
    "<video>",        # 8  (Ω 预留)
    "<audio>",        # 9  (Ω 预留)
    "<think>",        # 10
]
# 预留 11–31，避免将来扩展时重训词表
SPECIAL_ORDER += [f"<reserved_{i}>" for i in range(11, 32)]

BOS_ID, EOS_ID, PAD_ID, IMAGE_ID = 2, 3, 0, 7

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def resolve_corpus(path: str) -> str:
    """接受 jsonl 文件，或含 raw/corpus.jsonl 的目录。"""
    if os.path.isfile(path):
        return path
    cand = os.path.join(path, "raw", "corpus.jsonl")
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError(f"找不到语料: {path} 或 {cand}")


def read_length(corpus_dir: str):
    """从 manifest.json 读取文档总数，供进度条使用（可选）。"""
    base = corpus_dir[:-len("corpus.jsonl")] if corpus_dir.endswith("corpus.jsonl") else corpus_dir
    manifest = os.path.join(base, "manifest.json")
    if os.path.isfile(manifest):
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                m = json.load(f)
            return sum(v.get("docs", 0) for v in m.get("sources", {}).values())
        except Exception:
            pass
    return None


def iter_texts(corpus_path: str):
    """逐行解析 corpus.jsonl，yield 文本。优先 orjson 提速。"""
    try:
        import orjson as _json
    except ImportError:
        import json as _json
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except Exception:
                continue
            t = obj.get("text") or obj.get("content") or ""
            if t:
                yield t


def train_unigram(corpus_path: str, vocab_size: int, length=None):
    """用 tokenizers 训练 Unigram 词表。"""
    from tokenizers import Tokenizer, normalizers, pre_tokenizers, trainers, models

    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.ByteFallback(),
    ])
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_ORDER,
        unk_token="<unk>",
        byte_fallback=True,
        show_progress=True,
    )
    logger.info(f"开始 Unigram 训练: vocab_size={vocab_size}, 特殊token={len(SPECIAL_ORDER)}")
    tokenizer.train_from_iterator(iter_texts(corpus_path), trainer=trainer, length=length)
    return tokenizer


def export_hf(tokenizer, output_dir: str, model_max_length: int):
    """包装为 PreTrainedTokenizerFast 并导出完整 HF 分词器目录。"""
    from transformers import PreTrainedTokenizerFast

    os.makedirs(output_dir, exist_ok=True)
    # 先存一份原始 tokenizer.json（保底）
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        additional_special_tokens=SPECIAL_ORDER[4:],
        model_max_length=model_max_length,
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(output_dir)
    return fast


def sync_config(config_path: str, new_vocab: int):
    """自动同步 config.py: vocab_size(5处+from_hf) 与 bos/eos/image_token_id。备份 .bak。"""
    bak = config_path + ".bak"
    shutil.copy2(config_path, bak)
    with open(config_path, "r", encoding="utf-8") as f:
        txt = f.read()

    def sub(pat, rep):
        nonlocal txt
        txt, n = re.subn(pat, rep, txt)
        return n

    n_default = sub(r"vocab_size: int = \d+", f"vocab_size: int = {new_vocab}")
    n_preset = sub(r"vocab_size=\d+", f"vocab_size={new_vocab}")
    n_hf = sub(r'vocab_size",\s*\d+\)', f'vocab_size", {new_vocab})')
    n_bos = sub(r"bos_token_id: int = \d+", f"bos_token_id: int = {BOS_ID}")
    n_eos = sub(r"eos_token_id: int = \d+", f"eos_token_id: int = {EOS_ID}")
    n_img = sub(r"image_token_id: Optional\[int\] = None", f"image_token_id: Optional[int] = {IMAGE_ID}")
    pat_bos_hf = r'bos_token_id=getattr\(hf_config, "bos_token_id", \d+\)'
    pat_eos_hf = r'eos_token_id=getattr\(hf_config, "eos_token_id", \d+\)'
    n_bos_hf = sub(pat_bos_hf, f'bos_token_id=getattr(hf_config, "bos_token_id", {BOS_ID})')
    n_eos_hf = sub(pat_eos_hf, f'eos_token_id=getattr(hf_config, "eos_token_id", {EOS_ID})')

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(txt)

    logger.info(f"config.py 已同步 (备份: {bak})")
    logger.info(f"  vocab: default×{n_default} preset×{n_preset} from_hf×{n_hf}  (期望 1/4/1)")
    logger.info(f"  bos={BOS_ID}(×{n_bos}) eos={EOS_ID}(×{n_eos}) image={IMAGE_ID}(×{n_img}) "
                f"from_hf bos/eos×{n_bos_hf}/{n_eos_hf}")
    if n_default != 1 or n_preset != 4 or n_hf != 1:
        logger.warning("  ⚠ vocab_size 替换数量异常，请人工核对 config.py")


def parse_args():
    p = argparse.ArgumentParser(description="StarMoon-z1 分词器训练 (Unigram)")
    p.add_argument("--corpus", required=True, help="corpus.jsonl 文件，或含 raw/corpus.jsonl 的目录")
    p.add_argument("--output-dir", default="./starmoon-tokenizer", help="导出目录")
    p.add_argument("--vocab-size", type=int, default=150000, help="目标词表大小")
    p.add_argument("--model-max-length", type=int, default=32768, help="tokenizer model_max_length")
    p.add_argument("--config-path", default=None, help="config.py 路径(默认自动探测)")
    p.add_argument("--no-sync-config", action="store_true", help="不自动改写 config.py")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config_path or os.path.join(repo_root, "StarMoonZ1", "model", "config.py")

    corpus_path = resolve_corpus(args.corpus)
    logger.info(f"语料: {corpus_path}")
    length = read_length(corpus_path)
    if length:
        logger.info(f"文档总数(来自 manifest): {length:,}")

    try:
        from tokenizers import __version__ as tv
        from transformers import __version__ as tfv
        logger.info(f"tokenizers {tv} / transformers {tfv}")
    except ImportError as e:
        logger.error(f"缺少依赖: {e}\n请先: pip install tokenizers transformers")
        return

    # 训练
    tokenizer = train_unigram(corpus_path, args.vocab_size, length=length)

    # 导出
    fast = export_hf(tokenizer, args.output_dir, args.model_max_length)
    actual_vocab = len(fast)
    logger.info(f"训练完成，实际词表大小: {actual_vocab:,} (目标 {args.vocab_size:,})")

    # 冒烟校验
    sample = "你好，世界！def hello():\n    return 'StarMoon'"
    ids = fast.encode(sample)
    dec = fast.decode(ids, skip_special_tokens=False)
    logger.info(f"冒烟: encode({sample!r}) -> {len(ids)} tokens")
    logger.info(f"       decode 往返: {dec!r}")
    assert len(ids) > 0, "encode 失败"

    # 同步 config
    if not args.no_sync_config:
        if not os.path.isfile(config_path):
            logger.error(f"找不到 config.py: {config_path}（用 --config-path 指定）")
        else:
            sync_config(config_path, actual_vocab)

    # 多模态占位校验
    for ph in ["<image>", "<video>", "<audio>"]:
        assert fast.convert_tokens_to_ids(ph) != fast.unk_token_id, f"{ph} 未被注册为特殊 token"
    logger.info("多模态占位 <image>/<video>/<audio> 已注册 ✓")

    logger.info("=" * 60)
    logger.info(f"分词器已导出: {args.output_dir}")
    logger.info("下一步: 在 scripts/train_cpt.py 把 BASE_MODEL 指向该目录，跑通 1B 预训练冒烟")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
