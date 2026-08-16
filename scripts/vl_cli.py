#!/usr/bin/env python
"""
StarMoon-y1 多模态命令行工具
============================

子命令：
  encode   对图像目录批量编码（需 GPU + transformers + torch）
  ask      单图/多图问答（文本 + 图像 → 生成）

示例：
  python scripts/vl_cli.py ask --model ./checkpoints/starmoon-vl \\
      --image ./cat.jpg --prompt "描述这张图片" --max-new-tokens 256

说明：本脚本在解析参数时不加载 torch；实际运行需要 GPU 环境与已训练的多模态权重。
多模态代码位于独立包 `StarMoonY1`（仓库根/多模态放这里/StarMoonY1），与基座
`StarMoonZ1` 分离；本脚本通过 sys.path 同时挂载两者。
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MULTIMODAL_DIR = os.path.join(_REPO_ROOT, "多模态放这里")
# 挂载：StarMoonY1 包（多模态代码）与 StarMoonZ1 包（基座 Decoder）
sys.path.insert(0, _MULTIMODAL_DIR)
sys.path.insert(0, _REPO_ROOT)


def _load_runtime():
    """延迟加载 torch / 模型，避免在 --help 等场景就触发重型依赖"""
    import torch
    from StarMoonZ1.model.config import StarMoonZ1Config
    from StarMoonY1.model import StarMoonY1ForCausalLMWithVision
    from StarMoonY1.processor import StarMoonY1VLProcessor
    from PIL import Image
    return torch, StarMoonZ1Config, StarMoonY1ForCausalLMWithVision, StarMoonY1VLProcessor, Image


def cmd_ask(args):
    torch, Cfg, Model, Proc, Image = _load_runtime()
    model = Model.from_pretrained(args.model)
    model.eval()
    # tokenizer：从模型同目录的 tokenizer 或 transformers 默认
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model.prepare_for_vision(tok)  # R1 接线：注册 <image> + 扩展 embedding/lm_head
    proc = Proc(model.config, tok, vision_tower=model.vision_tower)
    imgs = [Image.open(p).convert("RGB") for p in args.image] if args.image else None
    feats = proc.process(args.prompt, images=imgs, return_labels=False)
    input_ids = feats["input_ids"].to("cuda")
    pixel_values = feats["pixel_values"].to("cuda") if feats["pixel_values"] is not None else None
    with torch.no_grad():
        out = model.generate_with_images(
            input_ids, pixel_values=pixel_values, image_grids=feats["image_grids"],
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty,
            eos_token_id=model.config.eos_token_id, do_sample=args.do_sample)
    print(tok.decode(out[0], skip_special_tokens=True))


def cmd_encode(args):
    torch, Cfg, Model, Proc, Image = _load_runtime()
    model = Model.from_pretrained(args.model)
    model.eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model.prepare_for_vision(tok)  # R1 接线
    proc = Proc(model.config, tok, vision_tower=model.vision_tower)
    files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))]
    feats_all = []
    with torch.no_grad():
        for p in files:
            img = Image.open(p).convert("RGB")
            f = proc.process("", images=[img])
            pv = f["pixel_values"].to("cuda")
            feat = model.encode_images(pv, f["image_grids"])
            feats_all.append(feat.cpu())
    out = torch.cat(feats_all, dim=0)
    torch.save(out, args.output)
    print(f"已编码 {len(files)} 张图像，视觉 token 形状 {tuple(out.shape)} → {args.output}")


def build_parser():
    p = argparse.ArgumentParser(description="StarMoon-y1 多模态命令行工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="单图/多图问答")
    a.add_argument("--model", required=True, help="多模态模型目录")
    a.add_argument("--image", nargs="*", default=[], help="图像路径（可多张）")
    a.add_argument("--prompt", required=True, help="文本提示（含 <image> 占位符）")
    a.add_argument("--max-new-tokens", type=int, default=256)
    a.add_argument("--temperature", type=float, default=0.7)
    a.add_argument("--top-p", type=float, default=0.9)
    a.add_argument("--top-k", type=int, default=50)
    a.add_argument("--repetition-penalty", type=float, default=1.0)
    a.add_argument("--do-sample", action="store_true")
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("encode", help="批量编码图像目录")
    e.add_argument("--model", required=True)
    e.add_argument("--image-dir", required=True)
    e.add_argument("--output", default="image_features.pt")
    e.set_defaults(func=cmd_encode)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
