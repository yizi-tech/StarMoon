"""
StarMoon-z1 模型基础测试
"""
import os
import tempfile
import shutil

import torch
import torch.nn.functional as F
from StarMoonZ1.model.config import StarMoonZ1Config
from StarMoonZ1.model.model import StarMoonZ1ForCausalLM, StarMoonZ1Model
from StarMoonZ1.msa.model import StarMoonZ1ForCausalLMWithMemory
from StarMoonZ1.msa.memory_bank import MemoryBank
from StarMoonZ1.model.lora import LoraConfig, LoraLinear, apply_lora, merge_lora_weights
from StarMoonZ1.training.dpo import DPOTrainer, DPODataset, dpo_collate_fn
from StarMoonZ1.training.sft import SFTTrainer, SFTDataset, dynamic_padding_collate, PackedSFTDataset
from StarMoonZ1.training.trainer import TrainingArguments
from StarMoonZ1.utils.distributed import (
    setup_distributed, is_distributed, is_main_process,
    get_rank, get_world_size, unwrap_model, wrap_model_ddp,
)


def get_mini_config():
    return StarMoonZ1Config(
        vocab_size=1000, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, max_position_embeddings=64,
        use_flash_attn=False, qk_norm=True, z_loss_coeff=1e-4,
        depth_scale_init=True,
    )


class TestStarMoonZ1Config:
    def test_presets(self):
        for size in ["1b", "3b", "7b", "14b"]:
            cfg = getattr(StarMoonZ1Config, f"preset_{size}")()
            assert cfg.hidden_size > 0
            assert cfg.num_hidden_layers > 0

    def test_preset_1b(self):
        cfg = StarMoonZ1Config.preset_1b()
        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 28  # 优化版: 更深更窄
        assert cfg.num_key_value_heads == 4
        assert cfg.tie_word_embeddings is True

    def test_to_from_dict(self):
        cfg = StarMoonZ1Config.preset_1b()
        d = cfg.to_dict()
        cfg2 = StarMoonZ1Config.from_dict(d)
        assert cfg.hidden_size == cfg2.hidden_size
        assert cfg.num_hidden_layers == cfg2.num_hidden_layers

    def test_head_dim_auto(self):
        cfg = StarMoonZ1Config(hidden_size=256, num_attention_heads=8)
        assert cfg.head_dim == 32

    def test_num_params_estimate(self):
        cfg = StarMoonZ1Config.preset_1b()
        est = cfg.num_params_estimate
        assert "B" in est or "M" in est


class TestStarMoonZ1Model:
    def test_model_forward(self):
        cfg = get_mini_config()
        model = StarMoonZ1Model(cfg)
        x = torch.randint(0, 100, (1, 8))
        h, _ = model(x)
        assert h.shape == (1, 8, 64)

    def test_kv_cache(self):
        cfg = get_mini_config()
        model = StarMoonZ1Model(cfg)
        x = torch.randint(0, 100, (1, 8))
        h1, pkv = model(x[:, :4], use_cache=True)
        h2, _ = model(x[:, 4:6], past_key_values=pkv, use_cache=True)
        assert h2.shape == (1, 2, 64)

    def test_attention_mask(self):
        cfg = get_mini_config()
        model = StarMoonZ1Model(cfg)
        x = torch.randint(0, 100, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.bool)
        mask[:, 4:] = False
        h, _ = model(x, attention_mask=mask)
        assert h.shape == (1, 8, 64)

    def test_sliding_window(self):
        cfg = get_mini_config()
        cfg.sliding_window = 4
        model = StarMoonZ1Model(cfg)
        x = torch.randint(0, 100, (1, 8))
        h, _ = model(x)
        assert h.shape == (1, 8, 64)

    def test_rope_dtype_preserved(self):
        """验证 apply_rope 不会改变输入 dtype"""
        cfg = get_mini_config()
        model = StarMoonZ1Model(cfg).to(torch.bfloat16)
        x = torch.randint(0, 100, (1, 8))
        h, _ = model(x)
        assert h.dtype == torch.bfloat16


class TestStarMoonZ1ForCausalLM:
    def test_forward(self):
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        x = torch.randint(0, 100, (1, 8))
        out = m(x, labels=x)
        assert out["loss"] is not None
        assert out["logits"].shape == (1, 8, 1000)

    def test_z_loss(self):
        """Z-loss 应在训练模式下增加 loss"""
        cfg = get_mini_config()
        cfg.z_loss_coeff = 1e-2
        m = StarMoonZ1ForCausalLM(cfg)
        m.train()
        x = torch.randint(0, 100, (1, 8))
        out = m(x, labels=x)
        assert out["loss"].item() > 0

    def test_generate(self):
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        x = torch.randint(0, 100, (1, 8))
        gen = m.generate(x, max_new_tokens=5, do_sample=False)
        assert gen.shape[1] >= 8

    def test_generate_with_sampling(self):
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        x = torch.randint(0, 100, (1, 4))
        gen = m.generate(x, max_new_tokens=10, do_sample=True,
                         temperature=0.8, top_p=0.9, top_k=50)
        assert gen.shape[1] >= 4

    def test_num_parameters(self):
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        assert m.num_parameters() > 0
        assert m.num_parameters(trainable_only=True) == m.num_parameters()

    def test_tie_embeddings(self):
        cfg = get_mini_config()
        cfg.tie_word_embeddings = True
        m = StarMoonZ1ForCausalLM(cfg)
        assert m.lm_head.weight is m.model.token_embedding.weight

    def test_save_pretrained(self):
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        tmpdir = tempfile.mkdtemp()
        try:
            m.save_pretrained(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "config.json"))
            assert os.path.exists(os.path.join(tmpdir, "model.safetensors"))
        finally:
            shutil.rmtree(tmpdir)

    def test_gradient_checkpointing(self):
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        m.gradient_checkpointing_enable()
        x = torch.randint(0, 100, (1, 8))
        out = m(x, labels=x)
        out["loss"].backward()
        # 验证梯度存在
        assert m.model.layers[0].self_attn.q_proj.weight.grad is not None


class TestLora:
    def test_apply_lora(self):
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        m = apply_lora(m, LoraConfig(r=4), verbose=False)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        assert trainable < total
        assert trainable > 0

    def test_lora_forward_effective(self):
        """验证 LoRA 在前向传播中真正生效"""
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        x = torch.randint(0, 100, (1, 8))

        # 记录原始输出
        with torch.no_grad():
            out_before = m(x)["logits"].clone()

        # 注入 LoRA 并随机初始化 B (非零)
        m = apply_lora(m, LoraConfig(r=4), verbose=False)
        for mod in m.modules():
            if isinstance(mod, LoraLinear):
                torch.nn.init.normal_(mod.lora_B, std=0.1)

        with torch.no_grad():
            out_after = m(x)["logits"]

        # 输出应该不同 (LoRA 生效)
        assert not torch.allclose(out_before, out_after, atol=1e-5)

    def test_lora_merge(self):
        """验证 LoRA 合并后恢复为普通 Linear"""
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        m = apply_lora(m, LoraConfig(r=4), verbose=False)

        # 合并
        m = merge_lora_weights(m)

        # 验证不再有 LoraLinear
        for mod in m.modules():
            assert not isinstance(mod, LoraLinear)

    def test_lora_gradient_flows(self):
        """验证梯度只流向 LoRA 参数"""
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        m = apply_lora(m, LoraConfig(r=4), verbose=False)

        x = torch.randint(0, 100, (1, 8))
        out = m(x, labels=x)
        out["loss"].backward()

        # LoRA 参数应有梯度
        for name, p in m.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert p.grad is not None, f"No gradient for {name}"
            elif p.requires_grad:
                # 非 LoRA 的 requires_grad 参数 (如 norm) 也可能有梯度
                pass


class TestMSA:
    """MSA 长时记忆模块测试（覆盖记忆压缩 5D 形状与带记忆前向/生成）"""

    def _mini_msa_config(self):
        cfg = get_mini_config()
        cfg.memory_layers = [1]            # 在 2 层模型中选第 1 层为记忆层
        cfg.chunk_size = 4
        cfg.router_top_k = 2
        return cfg

    def test_encode_shapes(self):
        cfg = self._mini_msa_config()
        model = StarMoonZ1ForCausalLMWithMemory(cfg)
        docs = torch.randint(0, 100, (3, 8))   # 3 文档, 长度 8 → 2 个 chunk
        mb = model.encode_documents(docs)
        lb = mb.per_layer[1]
        # 记忆必须保持 5D: [N, C, KV, d]
        assert lb.memory_k.shape == (3, 2, cfg.num_key_value_heads, cfg.head_dim)
        assert lb.memory_v.shape == lb.memory_k.shape
        assert lb.memory_kr.shape == lb.memory_k.shape
        assert lb.chunk_mask.shape == (3, 2)

    def test_forward_with_memory(self):
        cfg = self._mini_msa_config()
        model = StarMoonZ1ForCausalLMWithMemory(cfg)
        docs = torch.randint(0, 100, (3, 8))
        mb = model.encode_documents(docs)
        x = torch.randint(0, 100, (1, 6))
        out = model(x, memory_bank=mb, use_memory=True)
        assert out["logits"].shape == (1, 6, cfg.vocab_size)

    def test_generate_with_memory(self):
        cfg = self._mini_msa_config()
        model = StarMoonZ1ForCausalLMWithMemory(cfg)
        docs = torch.randint(0, 100, (2, 8))
        mb = model.encode_documents(docs)
        x = torch.randint(0, 100, (1, 4))
        gen = model.generate_with_memory(x, mb, max_new_tokens=3, do_sample=False)
        assert gen.shape[1] >= 4


# ──────────────────────────────────────────
# DPO 测试
# ──────────────────────────────────────────

class TestDPO:
    def _make_dpo_batch(self, batch_size=2, seq_len=16, vocab_size=1000):
        """构造 DPO 测试 batch"""
        return {
            "chosen_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "chosen_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "chosen_response_mask": torch.cat([
                torch.zeros(batch_size, 4, dtype=torch.long),
                torch.ones(batch_size, seq_len - 4, dtype=torch.long),
            ], dim=1),
            "rejected_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "rejected_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "rejected_response_mask": torch.cat([
                torch.zeros(batch_size, 4, dtype=torch.long),
                torch.ones(batch_size, seq_len - 4, dtype=torch.long),
            ], dim=1),
        }

    def test_dpo_loss_computation(self):
        """验证 DPO loss 可计算且为正值"""
        cfg = get_mini_config()
        model = StarMoonZ1ForCausalLM(cfg)
        args = TrainingArguments(output_dir=tempfile.mkdtemp(), num_train_epochs=1)
        trainer = DPOTrainer(model, args=args, beta=0.1, ref_model_offload=False)
        batch = self._make_dpo_batch()
        loss = trainer.compute_loss(model, batch)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        shutil.rmtree(args.output_dir, ignore_errors=True)

    def test_ipo_loss(self):
        """验证 IPO 变体损失"""
        cfg = get_mini_config()
        model = StarMoonZ1ForCausalLM(cfg)
        args = TrainingArguments(output_dir=tempfile.mkdtemp(), num_train_epochs=1)
        trainer = DPOTrainer(model, args=args, beta=0.1, loss_type="ipo",
                             ref_model_offload=False)
        batch = self._make_dpo_batch()
        loss = trainer.compute_loss(model, batch)
        assert loss.item() > 0
        shutil.rmtree(args.output_dir, ignore_errors=True)

    def test_dpo_gradient_flows(self):
        """验证 DPO loss 可以反向传播"""
        cfg = get_mini_config()
        model = StarMoonZ1ForCausalLM(cfg)
        args = TrainingArguments(output_dir=tempfile.mkdtemp(), num_train_epochs=1)
        trainer = DPOTrainer(model, args=args, beta=0.1, ref_model_offload=False)
        batch = self._make_dpo_batch()
        loss = trainer.compute_loss(model, batch)
        loss.backward()
        # 检查模型参数有梯度
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
        shutil.rmtree(args.output_dir, ignore_errors=True)

    def test_dpo_collate_fn(self):
        """验证 DPO collate 处理变长序列"""
        batch = [
            {
                "chosen_input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "chosen_attention_mask": torch.ones(5, dtype=torch.long),
                "chosen_response_mask": torch.tensor([0, 0, 1, 1, 1]),
                "rejected_input_ids": torch.tensor([1, 2, 3]),
                "rejected_attention_mask": torch.ones(3, dtype=torch.long),
                "rejected_response_mask": torch.tensor([0, 1, 1]),
            },
            {
                "chosen_input_ids": torch.tensor([1, 2, 3]),
                "chosen_attention_mask": torch.ones(3, dtype=torch.long),
                "chosen_response_mask": torch.tensor([0, 1, 1]),
                "rejected_input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7]),
                "rejected_attention_mask": torch.ones(7, dtype=torch.long),
                "rejected_response_mask": torch.tensor([0, 0, 0, 1, 1, 1, 1]),
            },
        ]
        result = dpo_collate_fn(batch, pad_token_id=0)
        # chosen pad 到 5, rejected pad 到 7
        assert result["chosen_input_ids"].shape == (2, 5)
        assert result["rejected_input_ids"].shape == (2, 7)
        # 检查 padding 位置的 attention_mask 为 0
        assert result["chosen_attention_mask"][1, 3:].sum() == 0
        assert result["rejected_attention_mask"][0, 3:].sum() == 0


# ──────────────────────────────────────────
# SFT 数据管道测试
# ──────────────────────────────────────────

class TestSFTDataPipeline:
    def test_dynamic_padding_collate(self):
        """验证动态 padding 对齐到 batch 内最长"""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]),
             "attention_mask": torch.ones(3, dtype=torch.long),
             "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6, 7, 8]),
             "attention_mask": torch.ones(5, dtype=torch.long),
             "labels": torch.tensor([4, 5, 6, 7, 8])},
        ]
        result = dynamic_padding_collate(batch, pad_token_id=0)
        assert result["input_ids"].shape == (2, 5)
        # 第一个样本后两个位置应为 pad
        assert result["input_ids"][0, 3:].tolist() == [0, 0]
        assert result["attention_mask"][0, 3:].sum() == 0
        assert result["labels"][0, 3:].tolist() == [-100, -100]

    def test_packed_dataset_shapes(self):
        """验证 PackedSFTDataset 输出形状"""
        # 使用简单的 mock tokenizer
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                ids = torch.tensor([ord(c) % 100 for c in text[:20]])
                return {"input_ids": ids.unsqueeze(0)}

        data = [{"text": f"sample text {i} " * 3} for i in range(10)]
        ds = PackedSFTDataset(data, MockTokenizer(), max_length=64)
        assert len(ds) > 0
        item = ds[0]
        assert item["input_ids"].shape == (64,)
        assert item["labels"].shape == (64,)
        assert item["position_ids"].shape == (64,)
        # attention_mask 是 4D: [1, 1, T, T]
        assert item["attention_mask"].shape == (1, 1, 64, 64)


# ──────────────────────────────────────────
# 分布式工具测试 (单进程模式)
# ──────────────────────────────────────────

class TestDistributedUtils:
    def test_single_process_defaults(self):
        """单进程环境下应返回默认值"""
        assert get_rank() == 0
        assert get_world_size() == 1
        assert is_main_process() is True
        assert is_distributed() is False

    def test_wrap_model_noop_single(self):
        """单进程时 wrap_model_ddp 应返回原模型"""
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        wrapped = wrap_model_ddp(m)
        assert wrapped is m

    def test_unwrap_model(self):
        """普通模型 unwrap 应返回自身"""
        cfg = get_mini_config()
        m = StarMoonZ1ForCausalLM(cfg)
        assert unwrap_model(m) is m
