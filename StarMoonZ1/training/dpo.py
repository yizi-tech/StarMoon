"""
DPO Trainer - 直接偏好优化训练器 (优化版)
支持: prompt label masking、IPO 变体、ref_model 显存优化、长度归一化。
"""
from __future__ import annotations
from typing import Optional, List, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from StarMoonZ1.training.trainer import TrainerBase, TrainingArguments


class DPOTrainer(TrainerBase):
    """
    直接偏好优化 (Direct Preference Optimization) 训练器。
    
    优化点:
    - prompt 部分不参与 loss 计算 (label masking)
    - 支持 DPO / IPO 两种损失函数
    - 长度归一化避免长序列偏好偏差
    - ref_model 延迟加载到 GPU，节省显存
    """
    def __init__(self, model, ref_model=None, args: TrainingArguments = None,
                 train_dataset: Optional[Dataset] = None,
                 beta: float = 0.1,
                 loss_type: str = "dpo",        # "dpo" | "ipo"
                 label_smoothing: float = 0.0,
                 length_normalize: bool = True,
                 ref_model_offload: bool = True):
        super().__init__(model, args, train_dataset)
        self.ref_model = ref_model or self._build_ref_model()
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.length_normalize = length_normalize
        self.ref_model_offload = ref_model_offload

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # 显存优化: ref_model 放在 CPU，需要时再移到 GPU
        if self.ref_model_offload and torch.cuda.is_available():
            self.ref_model.to("cpu")

    def _build_ref_model(self):
        import copy
        ref = copy.deepcopy(self.model)
        ref.eval()
        return ref

    def compute_loss(self, model, batch):
        chosen_ids = batch["chosen_input_ids"]
        rejected_ids = batch["rejected_input_ids"]
        chosen_mask = batch.get("chosen_attention_mask")
        rejected_mask = batch.get("rejected_attention_mask")
        # prompt 长度掩码: 1 = response 部分, 0 = prompt 部分
        chosen_response_mask = batch.get("chosen_response_mask")
        rejected_response_mask = batch.get("rejected_response_mask")

        # Policy model forward
        chosen_logits = model(input_ids=chosen_ids, attention_mask=chosen_mask)["logits"]
        rejected_logits = model(input_ids=rejected_ids, attention_mask=rejected_mask)["logits"]

        # Reference model forward (显存优化)
        if self.ref_model_offload:
            self.ref_model.to(self.device)

        ref_device = next(self.ref_model.parameters()).device

        with torch.no_grad():
            ref_chosen = self.ref_model(
                input_ids=chosen_ids.to(ref_device),
                attention_mask=chosen_mask.to(ref_device) if chosen_mask is not None else None,
            )["logits"].to(chosen_logits.device)
            ref_rejected = self.ref_model(
                input_ids=rejected_ids.to(ref_device),
                attention_mask=rejected_mask.to(ref_device) if rejected_mask is not None else None,
            )["logits"].to(rejected_logits.device)

        if self.ref_model_offload:
            self.ref_model.to("cpu")
            torch.cuda.empty_cache()

        # 计算 log probabilities
        pi_c = self._get_logps(chosen_logits, chosen_ids, chosen_response_mask)
        pi_r = self._get_logps(rejected_logits, rejected_ids, rejected_response_mask)
        rc = self._get_logps(ref_chosen, chosen_ids, chosen_response_mask)
        rr = self._get_logps(ref_rejected, rejected_ids, rejected_response_mask)

        # 损失计算
        log_ratio = (pi_c - pi_r) - (rc - rr)

        if self.loss_type == "ipo":
            # IPO: (log_ratio - 1/(2β))² — 更稳定，不易过拟合
            loss = (log_ratio - 1.0 / (2.0 * self.beta)).pow(2).mean()
        else:
            # 标准 DPO: -log σ(β · log_ratio)，带 label smoothing
            if self.label_smoothing > 0:
                loss = (
                    -self.label_smoothing * F.logsigmoid(-self.beta * log_ratio)
                    - (1 - self.label_smoothing) * F.logsigmoid(self.beta * log_ratio)
                ).mean()
            else:
                loss = -F.logsigmoid(self.beta * log_ratio).mean()

        return loss

    def _get_logps(self, logits: torch.Tensor, labels: torch.Tensor,
                   response_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算序列 log probabilities。
        - 仅在 response 部分计算 (prompt 部分被 mask)
        - 可选长度归一化
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        # 构建有效 token 掩码
        valid_mask = (shift_labels != -100).float()
        if response_mask is not None:
            # response_mask 对齐到 shift 后的长度
            resp = response_mask[..., 1:].contiguous().float()
            valid_mask = valid_mask * resp

        # 仅对有效 token 求和
        log_probs = -(per_token_loss * valid_mask).sum(dim=-1)

        # 长度归一化: 避免长序列天然有更大 log_prob
        if self.length_normalize:
            seq_len = valid_mask.sum(dim=-1).clamp(min=1)
            log_probs = log_probs / seq_len

        return log_probs


# ──────────────────────────────────────────
# DPO 数据集
# ──────────────────────────────────────────

class DPODataset(Dataset):
    """
    DPO 偏好数据集。
    
    数据格式:
    {
        "prompt": "用户问题",
        "chosen": "优选回答",
        "rejected": "劣选回答"
    }
    
    自动构建 response_mask 以屏蔽 prompt 部分。
    """
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_enc = self._encode_pair(prompt, chosen)
        rejected_enc = self._encode_pair(prompt, rejected)

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "chosen_response_mask": chosen_enc["response_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "rejected_response_mask": rejected_enc["response_mask"],
        }

    def _encode_pair(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        """编码 prompt+response，并标记 response 区域"""
        # 编码 prompt 部分
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=True,
                                       return_tensors="pt")["input_ids"].squeeze(0)
        # 编码 response 部分 (不加 special tokens)
        response_tokens = self.tokenizer(response, add_special_tokens=False,
                                         return_tensors="pt")["input_ids"].squeeze(0)
        # 添加 EOS
        eos = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        input_ids = torch.cat([prompt_tokens, response_tokens, eos], dim=0)

        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        attention_mask = torch.ones_like(input_ids)

        # response_mask: prompt 部分为 0, response 部分为 1
        response_mask = torch.zeros_like(input_ids)
        prompt_len = min(len(prompt_tokens), len(input_ids))
        response_mask[prompt_len:] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        }


# ──────────────────────────────────────────
# DPO 动态 Padding Collate
# ──────────────────────────────────────────

def dpo_collate_fn(batch: List[Dict[str, torch.Tensor]],
                   pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    DPO 专用 collate: 分别对 chosen/rejected 动态 padding。
    chosen 和 rejected 长度可能不同，分别 pad 到各自 batch 内最长。
    """
    # 分别计算 chosen / rejected 的最大长度
    max_chosen = max(item["chosen_input_ids"].size(0) for item in batch)
    max_rejected = max(item["rejected_input_ids"].size(0) for item in batch)

    chosen_ids, chosen_mask, chosen_resp = [], [], []
    rejected_ids, rejected_mask, rejected_resp = [], [], []

    for item in batch:
        # Pad chosen
        c_len = item["chosen_input_ids"].size(0)
        c_pad = max_chosen - c_len
        chosen_ids.append(torch.cat([item["chosen_input_ids"],
                                     torch.full((c_pad,), pad_token_id, dtype=torch.long)]))
        chosen_mask.append(torch.cat([item["chosen_attention_mask"],
                                      torch.zeros(c_pad, dtype=torch.long)]))
        chosen_resp.append(torch.cat([item["chosen_response_mask"],
                                      torch.zeros(c_pad, dtype=torch.long)]))
        # Pad rejected
        r_len = item["rejected_input_ids"].size(0)
        r_pad = max_rejected - r_len
        rejected_ids.append(torch.cat([item["rejected_input_ids"],
                                       torch.full((r_pad,), pad_token_id, dtype=torch.long)]))
        rejected_mask.append(torch.cat([item["rejected_attention_mask"],
                                        torch.zeros(r_pad, dtype=torch.long)]))
        rejected_resp.append(torch.cat([item["rejected_response_mask"],
                                        torch.zeros(r_pad, dtype=torch.long)]))

    return {
        "chosen_input_ids": torch.stack(chosen_ids),
        "chosen_attention_mask": torch.stack(chosen_mask),
        "chosen_response_mask": torch.stack(chosen_resp),
        "rejected_input_ids": torch.stack(rejected_ids),
        "rejected_attention_mask": torch.stack(rejected_mask),
        "rejected_response_mask": torch.stack(rejected_resp),
    }
