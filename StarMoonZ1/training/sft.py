"""
SFT Trainer - 监督微调训练器 (优化版)
支持: 指令感知 label masking、动态 padding、序列 packing、多轮对话格式。
"""
from __future__ import annotations
from functools import partial
from typing import Optional, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from StarMoonZ1.training.trainer import TrainerBase, TrainingArguments
from StarMoonZ1.utils.distributed import create_distributed_sampler


class SFTTrainer(TrainerBase):
    def __init__(self, model, args: TrainingArguments,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 pad_token_id: int = 0):
        super().__init__(model, args, train_dataset, eval_dataset)
        self.pad_token_id = pad_token_id

    def _create_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        """SFT 默认使用动态 padding collate，避免浪费算力"""
        sampler = create_distributed_sampler(dataset, shuffle=shuffle)
        kwargs = dict(
            batch_size=batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory and torch.cuda.is_available(),
            drop_last=shuffle,
            persistent_workers=self.args.dataloader_num_workers > 0,
            collate_fn=partial(dynamic_padding_collate, pad_token_id=self.pad_token_id),
        )
        if sampler is not None:
            kwargs["sampler"] = sampler
        else:
            kwargs["shuffle"] = shuffle
        if self.args.dataloader_num_workers > 0:
            kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(dataset, **kwargs)

    def compute_loss(self, model, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels", input_ids.clone())
        position_ids = batch.get("position_ids")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels, position_ids=position_ids)
        return outputs["loss"]


# ──────────────────────────────────────────
# 数据集: 指令感知 label masking
# ──────────────────────────────────────────

class SFTDataset(Dataset):
    """
    监督微调数据集。
    
    支持两种格式:
    1. 纯文本: {"text": "..."}
    2. 多轮对话: {"messages": [{"role": "system/user/assistant", "content": "..."}]}
    
    对多轮对话格式，仅在 assistant 回复部分计算 loss (指令感知 masking)。
    """
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048,
                 mask_instruction: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_instruction = mask_instruction

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if "messages" in item:
            return self._encode_chat(item["messages"])
        else:
            text = item.get("text", item.get("content", str(item)))
            return self._encode_text(text)

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

    def _encode_chat(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """多轮对话编码: 仅对 assistant 部分计算 loss。

        修复: 对整个 messages 列表应用 chat_template 一次以获得正确的整体
        tokenization，再通过前缀差分定位每个 assistant 回复的 token 区间，
        避免逐条 apply 导致特殊 token / 角色标记错乱 (如重复 BOS、turn 标记错位)。
        """
        if not self.mask_instruction:
            # 不 masking: 整体作为 labels
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            return self._encode_text(full_text)

        # 1) 整体 tokenization 得到完整 token 序列
        full_ids = list(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False))

        # 2) 前缀差分: 逐条前缀模板化，定位第 i 条 message 引入的 token 区间
        labels = [-100] * len(full_ids)
        try:
            prev_ids = list(self.tokenizer.apply_chat_template(
                [], tokenize=True, add_generation_prompt=False))
        except Exception:
            prev_ids = []
        for i in range(len(messages)):
            cur_ids = list(self.tokenizer.apply_chat_template(
                messages[:i + 1], tokenize=True, add_generation_prompt=False))
            span = cur_ids[len(prev_ids):]
            if messages[i].get("role") == "assistant":
                for off, tok in enumerate(span):
                    labels[len(prev_ids) + off] = tok
            prev_ids = cur_ids

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ──────────────────────────────────────────
# 动态 Padding Collate
# ──────────────────────────────────────────

def dynamic_padding_collate(batch: List[Dict[str, torch.Tensor]],
                            pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    动态 padding: 按 batch 内最长序列 padding，避免浪费算力。
    兼容 PackedSFTDataset: 其已预填充 position_ids 与 4D block-diagonal mask。
    """
    max_len = max(item["input_ids"].size(0) for item in batch)
    has_pos = "position_ids" in batch[0]

    input_ids_list, attn_mask_list, labels_list, pos_ids_list = [], [], [], []
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        input_ids_list.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attn = item["attention_mask"]
        if attn.dim() == 1:
            attn_mask_list.append(
                torch.cat([attn, torch.zeros(pad_len, dtype=torch.long)]))
        else:
            # 4D 预填充掩码 (PackedSFTDataset 已按 max_length 对齐，无需再 pad)
            attn_mask_list.append(attn)
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        if has_pos:
            p = item.get("position_ids", torch.zeros(seq_len, dtype=torch.long))
            pos_ids_list.append(
                torch.cat([p, torch.zeros(pad_len, dtype=torch.long)]))

    out = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_mask_list),
        "labels": torch.stack(labels_list),
    }
    if has_pos:
        out["position_ids"] = torch.stack(pos_ids_list)
    return out


# ──────────────────────────────────────────
# 序列 Packing (高效训练)
# ──────────────────────────────────────────

class PackedSFTDataset(Dataset):
    """
    序列 Packing: 将多条短样本拼接为一条长序列，最大化 GPU 利用率。
    使用 position_ids 重置 + block-diagonal attention mask 防止跨样本注意力。
    """
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.packed_samples = self._pack(data)

    def _pack(self, data: List[Dict]) -> List[Dict[str, torch.Tensor]]:
        packed = []
        current_ids, current_labels = [], []
        current_boundaries = []  # 记录每个子样本的起始位置
        current_len = 0

        for item in data:
            text = item.get("text", item.get("content", str(item)))
            tokens = self.tokenizer(text, add_special_tokens=True,
                                    return_tensors="pt")["input_ids"].squeeze(0)
            tok_len = len(tokens)

            if tok_len > self.max_length:
                tokens = tokens[:self.max_length]
                tok_len = self.max_length

            if current_len + tok_len > self.max_length:
                # 当前 pack 已满，保存并开始新 pack
                if current_len > 0:
                    packed.append(self._make_packed_item(
                        current_ids, current_labels, current_boundaries))
                current_ids, current_labels = [tokens], [tokens.clone()]
                current_boundaries = [0]
                current_len = tok_len
            else:
                if current_len == 0:
                    current_boundaries = [0]
                else:
                    current_boundaries.append(current_len)
                current_ids.append(tokens)
                current_labels.append(tokens.clone())
                current_len += tok_len

        # 最后一组
        if current_len > 0:
            packed.append(self._make_packed_item(
                current_ids, current_labels, current_boundaries))
        return packed

    def _make_packed_item(self, ids_list, labels_list, boundaries) -> Dict[str, torch.Tensor]:
        input_ids = torch.cat(ids_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        seq_len = len(input_ids)

        # 构建 position_ids: 每个子样本从 0 重新计数
        position_ids = torch.zeros(seq_len, dtype=torch.long)
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else seq_len
            position_ids[start:end] = torch.arange(end - start, dtype=torch.long)

        # 向量化构建 block-diagonal causal mask (4D: [1, 1, T, T])
        # 使用 segment_id 矩阵比较，避免 Python 循环，大序列下快 100x+
        segment_ids = torch.zeros(seq_len, dtype=torch.long)
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else seq_len
            segment_ids[start:end] = i

        # 因果性: col <= row
        row_idx = torch.arange(seq_len).unsqueeze(1)  # (T, 1)
        col_idx = torch.arange(seq_len).unsqueeze(0)  # (1, T)
        causal = col_idx <= row_idx
        # 同一段内才能互相注意
        same_segment = segment_ids.unsqueeze(1) == segment_ids.unsqueeze(0)
        # 组合: 因果 + 同段 => 可见，其余为 -inf
        visible = causal & same_segment
        attention_mask = torch.where(
            visible,
            torch.zeros(seq_len, seq_len),
            torch.full((seq_len, seq_len), float("-inf")),
        )

        # padding 到 max_length
        pad_len = self.max_length - seq_len
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=torch.long)])
            # 先 pad 列 (右侧): (seq_len, seq_len) -> (seq_len, max_length)
            right_pad = torch.full((seq_len, pad_len), float("-inf"))
            attention_mask = torch.cat([attention_mask, right_pad], dim=1)
            # 再 pad 行 (底部): (seq_len, max_length) -> (max_length, max_length)
            bottom_pad = torch.full((pad_len, self.max_length), float("-inf"))
            attention_mask = torch.cat([attention_mask, bottom_pad], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.unsqueeze(0).unsqueeze(0),  # [1, 1, T, T]
            "labels": labels,
            "position_ids": position_ids,
        }

    def __len__(self):
        return len(self.packed_samples)

    def __getitem__(self, idx):
        return self.packed_samples[idx]
