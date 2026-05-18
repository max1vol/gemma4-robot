from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


SNAC_CODEBOOK_SIZE = 4096
SAMPLE_RATE = 24_000
SNAC_LEVEL0_AUDIO_SAMPLES = 2048


@dataclass(frozen=True)
class StreamingConfig:
    first_level0_tokens: int = 20
    next_level0_tokens: int = 40
    overlap_level0_tokens: int = 10
    sample_rate: int = SAMPLE_RATE
    level0_audio_samples: int = SNAC_LEVEL0_AUDIO_SAMPLES


class SNACProjectionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_level_lengths: list[int],
        width: int = 512,
        codebook_size: int = SNAC_CODEBOOK_SIZE,
    ):
        super().__init__()
        self.max_level_lengths = max_level_lengths
        self.codebook_size = codebook_size
        self.cond = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.level_embed = nn.Embedding(len(max_level_lengths), width)
        self.pos_embed = nn.ModuleList(nn.Embedding(length, width) for length in max_level_lengths)
        self.mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, codebook_size),
        )

    def forward_level(self, cond_vectors: torch.Tensor, level: int, length: int | None = None) -> torch.Tensor:
        batch = cond_vectors.shape[0]
        if length is None:
            length = self.max_level_lengths[level]
        length = min(length, self.max_level_lengths[level])
        base = self.cond(cond_vectors).unsqueeze(1)
        positions = torch.arange(length, device=cond_vectors.device)
        pos = self.pos_embed[level](positions).unsqueeze(0)
        lvl = self.level_embed(torch.full((batch, length), level, device=cond_vectors.device))
        return self.mlp(base + pos + lvl)


class SNACSequenceHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_level_lengths: list[int],
        width: int = 512,
        layers: int = 2,
        codebook_size: int = SNAC_CODEBOOK_SIZE,
    ):
        super().__init__()
        self.max_level_lengths = max_level_lengths
        self.codebook_size = codebook_size
        self.bos_token_id = codebook_size
        self.layers = layers
        self.width = width
        self.cond = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.cond_to_hidden = nn.Linear(width, layers * width)
        self.token_embed = nn.Embedding(codebook_size + 1, width)
        self.level_embed = nn.Embedding(len(max_level_lengths), width)
        self.pos_embed = nn.ModuleList(nn.Embedding(length, width) for length in max_level_lengths)
        self.gru = nn.GRU(width, width, num_layers=layers, batch_first=True)
        self.out = nn.Linear(width, codebook_size)

    def _initial_hidden(self, cond_vectors: torch.Tensor) -> torch.Tensor:
        batch = cond_vectors.shape[0]
        cond = self.cond(cond_vectors)
        hidden = self.cond_to_hidden(cond).view(batch, self.layers, self.width)
        return hidden.transpose(0, 1).contiguous()

    def _step_inputs(self, tokens: torch.Tensor, level: int) -> torch.Tensor:
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        tok = self.token_embed(tokens.clamp(0, self.bos_token_id))
        pos = self.pos_embed[level](positions).unsqueeze(0)
        lvl = self.level_embed(torch.full((batch, length), level, device=tokens.device))
        return tok + pos + lvl

    def forward_level_teacher(
        self,
        cond_vectors: torch.Tensor,
        level: int,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        clean_targets = target_tokens.masked_fill(target_tokens < 0, 0)
        bos = torch.full(
            (target_tokens.shape[0], 1),
            self.bos_token_id,
            dtype=torch.long,
            device=target_tokens.device,
        )
        inputs = torch.cat([bos, clean_targets[:, :-1]], dim=1)
        x = self._step_inputs(inputs, level)
        out, _ = self.gru(x, self._initial_hidden(cond_vectors))
        return self.out(out)

    def generate_level(self, cond_vector: torch.Tensor, level: int, length: int) -> torch.Tensor:
        batch = cond_vector.shape[0]
        length = min(length, self.max_level_lengths[level])
        hidden = self._initial_hidden(cond_vector)
        token = torch.full(
            (batch, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=cond_vector.device,
        )
        generated = []
        for index in range(length):
            position = torch.tensor([index], device=cond_vector.device)
            x = (
                self.token_embed(token)
                + self.pos_embed[level](position).unsqueeze(0)
                + self.level_embed(torch.full((batch, 1), level, device=cond_vector.device))
            )
            out, hidden = self.gru(x, hidden)
            next_token = self.out(out[:, -1]).argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            token = next_token
        return torch.cat(generated, dim=1) if generated else torch.empty((batch, 0), device=cond_vector.device)


class SNACMemorySequenceHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_level_lengths: list[int],
        width: int = 512,
        layers: int = 2,
        heads: int = 4,
        codebook_size: int = SNAC_CODEBOOK_SIZE,
    ):
        super().__init__()
        self.max_level_lengths = max_level_lengths
        self.codebook_size = codebook_size
        self.bos_token_id = codebook_size
        self.layers = layers
        self.width = width
        self.memory = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.memory_to_hidden = nn.Linear(width, layers * width)
        self.token_embed = nn.Embedding(codebook_size + 1, width)
        self.level_embed = nn.Embedding(len(max_level_lengths), width)
        self.pos_embed = nn.ModuleList(nn.Embedding(length, width) for length in max_level_lengths)
        self.gru = nn.GRU(width, width, num_layers=layers, batch_first=True)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, codebook_size)

    def _project_memory(self, memory_vectors: torch.Tensor) -> torch.Tensor:
        return self.memory(memory_vectors)

    def _pooled_memory(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        weights = memory_mask.float().unsqueeze(-1)
        return (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _initial_hidden(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        batch = memory.shape[0]
        pooled = self._pooled_memory(memory, memory_mask)
        hidden = self.memory_to_hidden(pooled).view(batch, self.layers, self.width)
        return hidden.transpose(0, 1).contiguous()

    def _step_inputs(self, tokens: torch.Tensor, level: int) -> torch.Tensor:
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        tok = self.token_embed(tokens.clamp(0, self.bos_token_id))
        pos = self.pos_embed[level](positions).unsqueeze(0)
        lvl = self.level_embed(torch.full((batch, length), level, device=tokens.device))
        return tok + pos + lvl

    def forward_level_teacher(
        self,
        memory_vectors: torch.Tensor,
        memory_mask: torch.Tensor,
        level: int,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        memory = self._project_memory(memory_vectors)
        clean_targets = target_tokens.masked_fill(target_tokens < 0, 0)
        bos = torch.full(
            (target_tokens.shape[0], 1),
            self.bos_token_id,
            dtype=torch.long,
            device=target_tokens.device,
        )
        inputs = torch.cat([bos, clean_targets[:, :-1]], dim=1)
        x = self._step_inputs(inputs, level)
        out, _ = self.gru(x, self._initial_hidden(memory, memory_mask))
        attended, _ = self.attn(out, memory, memory, key_padding_mask=~memory_mask)
        return self.out(self.norm(out + attended))

    def generate_level(
        self,
        memory_vectors: torch.Tensor,
        memory_mask: torch.Tensor,
        level: int,
        length: int,
    ) -> torch.Tensor:
        batch = memory_vectors.shape[0]
        length = min(length, self.max_level_lengths[level])
        memory = self._project_memory(memory_vectors)
        hidden = self._initial_hidden(memory, memory_mask)
        token = torch.full(
            (batch, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=memory_vectors.device,
        )
        generated = []
        for index in range(length):
            position = torch.tensor([index], device=memory_vectors.device)
            x = (
                self.token_embed(token)
                + self.pos_embed[level](position).unsqueeze(0)
                + self.level_embed(torch.full((batch, 1), level, device=memory_vectors.device))
            )
            out, hidden = self.gru(x, hidden)
            attended, _ = self.attn(out, memory, memory, key_padding_mask=~memory_mask)
            next_token = self.out(self.norm(out[:, -1] + attended[:, -1])).argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            token = next_token
        return torch.cat(generated, dim=1) if generated else torch.empty((batch, 0), device=memory_vectors.device)


class SNACTextScaffoldSequenceHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_level_lengths: list[int],
        width: int = 512,
        layers: int = 2,
        heads: int = 4,
        codebook_size: int = SNAC_CODEBOOK_SIZE,
    ):
        super().__init__()
        self.max_level_lengths = max_level_lengths
        self.codebook_size = codebook_size
        self.bos_token_id = codebook_size
        self.layers = layers
        self.width = width
        self.style = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.text = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.context_to_hidden = nn.Linear(width * 2, layers * width)
        self.token_embed = nn.Embedding(codebook_size + 1, width)
        self.level_embed = nn.Embedding(len(max_level_lengths), width)
        self.pos_embed = nn.ModuleList(nn.Embedding(length, width) for length in max_level_lengths)
        self.gru = nn.GRU(width, width, num_layers=layers, batch_first=True)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, codebook_size)

    def _project_text(self, text_vectors: torch.Tensor) -> torch.Tensor:
        return self.text(text_vectors)

    def _pooled_text(self, text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        weights = text_mask.float().unsqueeze(-1)
        return (text * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _initial_hidden(
        self,
        style_vectors: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = style_vectors.shape[0]
        style = self.style(style_vectors)
        pooled_text = self._pooled_text(text, text_mask)
        hidden = self.context_to_hidden(torch.cat([style, pooled_text], dim=-1)).view(batch, self.layers, self.width)
        return hidden.transpose(0, 1).contiguous()

    def _aligned_text(self, text: torch.Tensor, text_mask: torch.Tensor, length: int) -> torch.Tensor:
        batch, _, width = text.shape
        valid_lengths = text_mask.sum(dim=1).clamp_min(1)
        if length <= 0:
            return torch.empty((batch, 0, width), dtype=text.dtype, device=text.device)
        positions = torch.arange(length, device=text.device, dtype=torch.float32).unsqueeze(0)
        denominator = max(length - 1, 1)
        max_text_index = (valid_lengths - 1).float().unsqueeze(1)
        indices = torch.round((positions / denominator) * max_text_index).long()
        indices = indices.clamp_min(0)
        gather_index = indices.unsqueeze(-1).expand(batch, length, width)
        return text.gather(dim=1, index=gather_index)

    def _step_inputs(
        self,
        tokens: torch.Tensor,
        level: int,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        tok = self.token_embed(tokens.clamp(0, self.bos_token_id))
        pos = self.pos_embed[level](positions).unsqueeze(0)
        lvl = self.level_embed(torch.full((batch, length), level, device=tokens.device))
        aligned = self._aligned_text(text, text_mask, length)
        return tok + pos + lvl + aligned

    def forward_level_teacher(
        self,
        style_vectors: torch.Tensor,
        text_vectors: torch.Tensor,
        text_mask: torch.Tensor,
        level: int,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        text = self._project_text(text_vectors)
        clean_targets = target_tokens.masked_fill(target_tokens < 0, 0)
        bos = torch.full(
            (target_tokens.shape[0], 1),
            self.bos_token_id,
            dtype=torch.long,
            device=target_tokens.device,
        )
        inputs = torch.cat([bos, clean_targets[:, :-1]], dim=1)
        x = self._step_inputs(inputs, level, text, text_mask)
        out, _ = self.gru(x, self._initial_hidden(style_vectors, text, text_mask))
        attended, _ = self.attn(out, text, text, key_padding_mask=~text_mask)
        return self.out(self.norm(out + attended))

    def forward_level_free_running(
        self,
        style_vectors: torch.Tensor,
        text_vectors: torch.Tensor,
        text_mask: torch.Tensor,
        level: int,
        length: int,
    ) -> torch.Tensor:
        batch = style_vectors.shape[0]
        length = min(length, self.max_level_lengths[level])
        text = self._project_text(text_vectors)
        hidden = self._initial_hidden(style_vectors, text, text_mask)
        aligned_text = self._aligned_text(text, text_mask, length)
        token = torch.full(
            (batch, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=style_vectors.device,
        )
        logits_by_step = []
        for index in range(length):
            position = torch.tensor([index], device=style_vectors.device)
            x = (
                self.token_embed(token)
                + self.pos_embed[level](position).unsqueeze(0)
                + self.level_embed(torch.full((batch, 1), level, device=style_vectors.device))
                + aligned_text[:, index : index + 1]
            )
            out, hidden = self.gru(x, hidden)
            attended, _ = self.attn(out, text, text, key_padding_mask=~text_mask)
            logits = self.out(self.norm(out[:, -1] + attended[:, -1]))
            logits_by_step.append(logits.unsqueeze(1))
            token = logits.argmax(dim=-1, keepdim=True).detach()
        if not logits_by_step:
            return torch.empty((batch, 0, self.codebook_size), device=style_vectors.device)
        return torch.cat(logits_by_step, dim=1)

    def generate_level(
        self,
        style_vectors: torch.Tensor,
        text_vectors: torch.Tensor,
        text_mask: torch.Tensor,
        level: int,
        length: int,
    ) -> torch.Tensor:
        batch = style_vectors.shape[0]
        length = min(length, self.max_level_lengths[level])
        text = self._project_text(text_vectors)
        hidden = self._initial_hidden(style_vectors, text, text_mask)
        aligned_text = self._aligned_text(text, text_mask, length)
        token = torch.full(
            (batch, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=style_vectors.device,
        )
        generated = []
        for index in range(length):
            position = torch.tensor([index], device=style_vectors.device)
            x = (
                self.token_embed(token)
                + self.pos_embed[level](position).unsqueeze(0)
                + self.level_embed(torch.full((batch, 1), level, device=style_vectors.device))
                + aligned_text[:, index : index + 1]
            )
            out, hidden = self.gru(x, hidden)
            attended, _ = self.attn(out, text, text, key_padding_mask=~text_mask)
            next_token = self.out(self.norm(out[:, -1] + attended[:, -1])).argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            token = next_token
        return torch.cat(generated, dim=1) if generated else torch.empty((batch, 0), device=style_vectors.device)


class SNACTextScaffoldProjectionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_level_lengths: list[int],
        width: int = 512,
        codebook_size: int = SNAC_CODEBOOK_SIZE,
    ):
        super().__init__()
        self.max_level_lengths = max_level_lengths
        self.codebook_size = codebook_size
        self.style = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.text = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.level_embed = nn.Embedding(len(max_level_lengths), width)
        self.pos_embed = nn.ModuleList(nn.Embedding(length, width) for length in max_level_lengths)
        self.mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, codebook_size),
        )

    def _project_text(self, text_vectors: torch.Tensor) -> torch.Tensor:
        return self.text(text_vectors)

    def _aligned_text(self, text: torch.Tensor, text_mask: torch.Tensor, length: int) -> torch.Tensor:
        batch, _, width = text.shape
        valid_lengths = text_mask.sum(dim=1).clamp_min(1)
        if length <= 0:
            return torch.empty((batch, 0, width), dtype=text.dtype, device=text.device)
        positions = torch.arange(length, device=text.device, dtype=torch.float32).unsqueeze(0)
        denominator = max(length - 1, 1)
        max_text_index = (valid_lengths - 1).float().unsqueeze(1)
        indices = torch.round((positions / denominator) * max_text_index).long().clamp_min(0)
        gather_index = indices.unsqueeze(-1).expand(batch, length, width)
        return text.gather(dim=1, index=gather_index)

    def forward_level(
        self,
        style_vectors: torch.Tensor,
        text_vectors: torch.Tensor,
        text_mask: torch.Tensor,
        level: int,
        length: int | None = None,
    ) -> torch.Tensor:
        batch = style_vectors.shape[0]
        if length is None:
            length = self.max_level_lengths[level]
        length = min(length, self.max_level_lengths[level])
        style = self.style(style_vectors).unsqueeze(1)
        text = self._project_text(text_vectors)
        aligned_text = self._aligned_text(text, text_mask, length)
        positions = torch.arange(length, device=style_vectors.device)
        pos = self.pos_embed[level](positions).unsqueeze(0)
        lvl = self.level_embed(torch.full((batch, length), level, device=style_vectors.device))
        return self.mlp(style + aligned_text + pos + lvl)


class SNACTextCoarseToFineSequenceHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_level_lengths: list[int],
        width: int = 512,
        layers: int = 2,
        heads: int = 4,
        codebook_size: int = SNAC_CODEBOOK_SIZE,
    ):
        super().__init__()
        self.max_level_lengths = max_level_lengths
        self.codebook_size = codebook_size
        self.bos_token_id = codebook_size
        self.layers = layers
        self.width = width
        self.style = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.text = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.context_to_hidden = nn.Linear(width * 3, layers * width)
        self.token_embed = nn.Embedding(codebook_size + 1, width)
        self.coarse_token_embed = nn.Embedding(codebook_size, width)
        self.level_embed = nn.Embedding(len(max_level_lengths), width)
        self.pos_embed = nn.ModuleList(nn.Embedding(length, width) for length in max_level_lengths)
        self.gru = nn.GRU(width, width, num_layers=layers, batch_first=True)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, codebook_size)

    def _project_text(self, text_vectors: torch.Tensor) -> torch.Tensor:
        return self.text(text_vectors)

    def _pooled_text(self, text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        weights = text_mask.float().unsqueeze(-1)
        return (text * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _aligned_text(self, text: torch.Tensor, text_mask: torch.Tensor, length: int) -> torch.Tensor:
        batch, _, width = text.shape
        valid_lengths = text_mask.sum(dim=1).clamp_min(1)
        if length <= 0:
            return torch.empty((batch, 0, width), dtype=text.dtype, device=text.device)
        positions = torch.arange(length, device=text.device, dtype=torch.float32).unsqueeze(0)
        denominator = max(length - 1, 1)
        max_text_index = (valid_lengths - 1).float().unsqueeze(1)
        indices = torch.round((positions / denominator) * max_text_index).long().clamp_min(0)
        gather_index = indices.unsqueeze(-1).expand(batch, length, width)
        return text.gather(dim=1, index=gather_index)

    def _coarse_context(self, coarse_tokens: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        if coarse_tokens is None or coarse_tokens.shape[1] == 0:
            return torch.zeros((batch, self.width), dtype=self.style[0].weight.dtype, device=device)
        clean = coarse_tokens.masked_fill(coarse_tokens < 0, 0).clamp(0, self.codebook_size - 1)
        return self.coarse_token_embed(clean).mean(dim=1)

    def _aligned_coarse(
        self,
        coarse_tokens: torch.Tensor | None,
        length: int,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if length <= 0:
            return torch.empty((batch, 0, self.width), dtype=self.style[0].weight.dtype, device=device)
        if coarse_tokens is None or coarse_tokens.shape[1] == 0:
            return torch.zeros((batch, length, self.width), dtype=self.style[0].weight.dtype, device=device)
        clean = coarse_tokens.masked_fill(coarse_tokens < 0, 0).clamp(0, self.codebook_size - 1)
        coarse_len = clean.shape[1]
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(0)
        denominator = max(length - 1, 1)
        max_coarse_index = coarse_len - 1
        indices = torch.round((positions / denominator) * max_coarse_index).long().expand(batch, length)
        aligned_tokens = clean.gather(dim=1, index=indices)
        return self.coarse_token_embed(aligned_tokens)

    def _initial_hidden(
        self,
        style_vectors: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        coarse_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = style_vectors.shape[0]
        style = self.style(style_vectors)
        pooled_text = self._pooled_text(text, text_mask)
        pooled_coarse = self._coarse_context(coarse_tokens, batch, style_vectors.device)
        hidden = self.context_to_hidden(torch.cat([style, pooled_text, pooled_coarse], dim=-1))
        hidden = hidden.view(batch, self.layers, self.width)
        return hidden.transpose(0, 1).contiguous()

    def _step_inputs(
        self,
        tokens: torch.Tensor,
        level: int,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        coarse_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        tok = self.token_embed(tokens.clamp(0, self.bos_token_id))
        pos = self.pos_embed[level](positions).unsqueeze(0)
        lvl = self.level_embed(torch.full((batch, length), level, device=tokens.device))
        aligned_text = self._aligned_text(text, text_mask, length)
        aligned_coarse = self._aligned_coarse(coarse_tokens, length, batch, tokens.device)
        return tok + pos + lvl + aligned_text + aligned_coarse

    def forward_level_teacher(
        self,
        style_vectors: torch.Tensor,
        text_vectors: torch.Tensor,
        text_mask: torch.Tensor,
        level: int,
        target_tokens: torch.Tensor,
        coarse_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text = self._project_text(text_vectors)
        clean_targets = target_tokens.masked_fill(target_tokens < 0, 0)
        bos = torch.full(
            (target_tokens.shape[0], 1),
            self.bos_token_id,
            dtype=torch.long,
            device=target_tokens.device,
        )
        inputs = torch.cat([bos, clean_targets[:, :-1]], dim=1)
        x = self._step_inputs(inputs, level, text, text_mask, coarse_tokens)
        out, _ = self.gru(x, self._initial_hidden(style_vectors, text, text_mask, coarse_tokens))
        attended, _ = self.attn(out, text, text, key_padding_mask=~text_mask)
        return self.out(self.norm(out + attended))

    def generate_level(
        self,
        style_vectors: torch.Tensor,
        text_vectors: torch.Tensor,
        text_mask: torch.Tensor,
        level: int,
        length: int,
        coarse_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = style_vectors.shape[0]
        length = min(length, self.max_level_lengths[level])
        text = self._project_text(text_vectors)
        hidden = self._initial_hidden(style_vectors, text, text_mask, coarse_tokens)
        aligned_text = self._aligned_text(text, text_mask, length)
        aligned_coarse = self._aligned_coarse(coarse_tokens, length, batch, style_vectors.device)
        token = torch.full(
            (batch, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=style_vectors.device,
        )
        generated = []
        for index in range(length):
            position = torch.tensor([index], device=style_vectors.device)
            x = (
                self.token_embed(token)
                + self.pos_embed[level](position).unsqueeze(0)
                + self.level_embed(torch.full((batch, 1), level, device=style_vectors.device))
                + aligned_text[:, index : index + 1]
                + aligned_coarse[:, index : index + 1]
            )
            out, hidden = self.gru(x, hidden)
            attended, _ = self.attn(out, text, text, key_padding_mask=~text_mask)
            next_token = self.out(self.norm(out[:, -1] + attended[:, -1])).argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            token = next_token
        return torch.cat(generated, dim=1) if generated else torch.empty((batch, 0), device=style_vectors.device)


def level0_stream_windows(total_level0_tokens: int, config: StreamingConfig) -> list[dict]:
    windows = []
    emitted_until = 0
    start = 0
    length = config.first_level0_tokens
    while start < total_level0_tokens:
        end = min(total_level0_tokens, start + length)
        trim_prefix_tokens = max(0, emitted_until - start)
        windows.append(
            {
                "index": len(windows),
                "start_level0": start,
                "end_level0": end,
                "level0_tokens": end - start,
                "trim_prefix_level0_tokens": trim_prefix_tokens,
            }
        )
        emitted_until = max(emitted_until, end)
        if end >= total_level0_tokens:
            break
        start = max(0, emitted_until - config.overlap_level0_tokens)
        length = config.next_level0_tokens
    return windows


def simulate_playback_queue(chunks: list[dict]) -> dict:
    scheduled_chunks = []
    playback_cursor_s = 0.0
    underrun_count = 0
    underrun_s = 0.0
    min_deadline_slack_ms = None
    max_queued_audio_s = 0.0

    for chunk in chunks:
        ready_s = chunk["cumulative_ready_ms"] / 1000.0
        queued_audio_s = max(0.0, playback_cursor_s - ready_s)
        max_queued_audio_s = max(max_queued_audio_s, queued_audio_s)
        if chunk["index"] == 0:
            play_start_s = ready_s
        elif ready_s > playback_cursor_s:
            underrun_count += 1
            underrun_s += ready_s - playback_cursor_s
            play_start_s = ready_s
        else:
            play_start_s = playback_cursor_s

        deadline_slack_ms = (play_start_s - ready_s) * 1000.0
        min_deadline_slack_ms = (
            deadline_slack_ms
            if min_deadline_slack_ms is None
            else min(min_deadline_slack_ms, deadline_slack_ms)
        )
        play_end_s = play_start_s + chunk["emitted_audio_s"]
        playback_cursor_s = play_end_s
        scheduled_chunks.append(
            {
                "index": chunk["index"],
                "decode_ready_s": ready_s,
                "play_start_s": play_start_s,
                "play_end_s": play_end_s,
                "emitted_audio_s": chunk["emitted_audio_s"],
                "queued_audio_s_when_ready": queued_audio_s,
                "deadline_slack_ms": deadline_slack_ms,
            }
        )

    return {
        "first_audio_ready_ms": chunks[0]["cumulative_ready_ms"] if chunks else 0.0,
        "first_audio_start_ms": scheduled_chunks[0]["play_start_s"] * 1000.0 if scheduled_chunks else 0.0,
        "total_playback_s": playback_cursor_s,
        "underrun_count": underrun_count,
        "underrun_s": underrun_s,
        "min_deadline_slack_ms": min_deadline_slack_ms or 0.0,
        "max_queued_audio_s": max_queued_audio_s,
        "scheduled_chunks": scheduled_chunks,
    }


def predict_snac_codes(
    head: SNACProjectionHead,
    cond_vector: torch.Tensor,
    snac_code_shapes: list[list[int]],
) -> tuple[list[torch.Tensor], dict, list[list[int]]]:
    decode_codes = []
    level_token_counts = {}
    predicted_values_by_level = []
    device = cond_vector.device
    with torch.inference_mode():
        for level, shape in enumerate(snac_code_shapes):
            required = int(np.prod(shape))
            logits = head.forward_level(cond_vector, level, length=required)
            pred = logits.argmax(dim=-1)[0, :required].contiguous()
            predicted_values_by_level.append(pred.detach().cpu().tolist())
            level_token_counts[level] = int(pred.numel())
            decode_codes.append(pred.to(device).view(*shape))
    return decode_codes, level_token_counts, predicted_values_by_level


def predict_snac_codes_sequence(
    head: SNACSequenceHead,
    cond_vector: torch.Tensor,
    snac_code_shapes: list[list[int]],
) -> tuple[list[torch.Tensor], dict, list[list[int]]]:
    decode_codes = []
    level_token_counts = {}
    predicted_values_by_level = []
    device = cond_vector.device
    with torch.inference_mode():
        for level, shape in enumerate(snac_code_shapes):
            required = int(np.prod(shape))
            pred = head.generate_level(cond_vector, level, required)[0, :required].contiguous()
            predicted_values_by_level.append(pred.detach().cpu().tolist())
            level_token_counts[level] = int(pred.numel())
            decode_codes.append(pred.to(device).view(*shape))
    return decode_codes, level_token_counts, predicted_values_by_level


def predict_snac_codes_memory(
    head: SNACMemorySequenceHead,
    memory_vectors: torch.Tensor,
    memory_mask: torch.Tensor,
    snac_code_shapes: list[list[int]],
) -> tuple[list[torch.Tensor], dict, list[list[int]]]:
    decode_codes = []
    level_token_counts = {}
    predicted_values_by_level = []
    device = memory_vectors.device
    with torch.inference_mode():
        for level, shape in enumerate(snac_code_shapes):
            required = int(np.prod(shape))
            pred = head.generate_level(memory_vectors, memory_mask, level, required)[0, :required].contiguous()
            predicted_values_by_level.append(pred.detach().cpu().tolist())
            level_token_counts[level] = int(pred.numel())
            decode_codes.append(pred.to(device).view(*shape))
    return decode_codes, level_token_counts, predicted_values_by_level


def predict_snac_codes_text_scaffold(
    head: SNACTextScaffoldSequenceHead,
    style_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    text_mask: torch.Tensor,
    snac_code_shapes: list[list[int]],
) -> tuple[list[torch.Tensor], dict, list[list[int]]]:
    decode_codes = []
    level_token_counts = {}
    predicted_values_by_level = []
    device = style_vectors.device
    with torch.inference_mode():
        for level, shape in enumerate(snac_code_shapes):
            required = int(np.prod(shape))
            pred = head.generate_level(style_vectors, text_vectors, text_mask, level, required)[0, :required].contiguous()
            predicted_values_by_level.append(pred.detach().cpu().tolist())
            level_token_counts[level] = int(pred.numel())
            decode_codes.append(pred.to(device).view(*shape))
    return decode_codes, level_token_counts, predicted_values_by_level


def predict_snac_codes_text_scaffold_projection(
    head: SNACTextScaffoldProjectionHead,
    style_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    text_mask: torch.Tensor,
    snac_code_shapes: list[list[int]],
) -> tuple[list[torch.Tensor], dict, list[list[int]]]:
    decode_codes = []
    level_token_counts = {}
    predicted_values_by_level = []
    device = style_vectors.device
    with torch.inference_mode():
        for level, shape in enumerate(snac_code_shapes):
            required = int(np.prod(shape))
            logits = head.forward_level(style_vectors, text_vectors, text_mask, level, length=required)
            pred = logits.argmax(dim=-1)[0, :required].contiguous()
            predicted_values_by_level.append(pred.detach().cpu().tolist())
            level_token_counts[level] = int(pred.numel())
            decode_codes.append(pred.to(device).view(*shape))
    return decode_codes, level_token_counts, predicted_values_by_level


def predict_snac_codes_text_coarse_to_fine(
    head: SNACTextCoarseToFineSequenceHead,
    style_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    text_mask: torch.Tensor,
    snac_code_shapes: list[list[int]],
) -> tuple[list[torch.Tensor], dict, list[list[int]]]:
    decode_codes = []
    level_token_counts = {}
    predicted_values_by_level = []
    generated_flat: list[torch.Tensor] = []
    device = style_vectors.device
    with torch.inference_mode():
        for level, shape in enumerate(snac_code_shapes):
            required = int(np.prod(shape))
            coarse_tokens = generated_flat[level - 1] if level > 0 else None
            pred = head.generate_level(
                style_vectors,
                text_vectors,
                text_mask,
                level,
                required,
                coarse_tokens=coarse_tokens,
            )[0, :required].contiguous()
            generated_flat.append(pred.unsqueeze(0))
            predicted_values_by_level.append(pred.detach().cpu().tolist())
            level_token_counts[level] = int(pred.numel())
            decode_codes.append(pred.to(device).view(*shape))
    return decode_codes, level_token_counts, predicted_values_by_level


def generate_streaming_chunks(
    head: SNACProjectionHead,
    cond_vector: torch.Tensor,
    snac_code_shapes: list[list[int]],
    decode_codes_to_audio: Callable[[list[torch.Tensor]], tuple[np.ndarray, float]],
    wav_bytes_from_array: Callable[[np.ndarray], bytes],
    config: StreamingConfig,
    perf_counter: Callable[[], float],
) -> tuple[list[dict], np.ndarray, dict]:
    level0_total = int(np.prod(snac_code_shapes[0]))
    windows = level0_stream_windows(level0_total, config)
    chunks = []
    stitched_parts = []
    cumulative_ready_ms = 0.0
    device = cond_vector.device

    for window in windows:
        chunk_started = perf_counter()
        chunk_codes = []
        chunk_level_token_counts = {}
        with torch.inference_mode():
            for level, shape in enumerate(snac_code_shapes):
                level_total = int(np.prod(shape))
                ratio = max(1, level_total // max(level0_total, 1))
                level_start = min(level_total, window["start_level0"] * ratio)
                level_end = min(level_total, window["end_level0"] * ratio)
                required = max(0, level_end - level_start)
                logits = head.forward_level(cond_vector, level, length=level_end)
                pred = logits.argmax(dim=-1)[0, level_start:level_end].contiguous()
                chunk_level_token_counts[level] = int(pred.numel())
                stream_shape = list(shape)
                stream_shape[-1] = required
                chunk_codes.append(pred.to(device).view(*stream_shape))
        chunk_head_ms = (perf_counter() - chunk_started) * 1000
        chunk_audio, chunk_decode_ms = decode_codes_to_audio(chunk_codes)
        trim_samples = min(
            len(chunk_audio),
            window["trim_prefix_level0_tokens"] * config.level0_audio_samples,
        )
        emitted_audio = chunk_audio[trim_samples:]
        stitched_parts.append(emitted_audio)
        chunk_wav_bytes = wav_bytes_from_array(chunk_audio)
        ready_ms = chunk_head_ms + chunk_decode_ms
        cumulative_ready_ms += ready_ms
        chunks.append(
            {
                **window,
                "level_token_counts": chunk_level_token_counts,
                "chunk_audio_s": len(chunk_audio) / config.sample_rate,
                "trim_prefix_audio_s": trim_samples / config.sample_rate,
                "emitted_audio_s": len(emitted_audio) / config.sample_rate,
                "head_ms": chunk_head_ms,
                "snac_decode_ms": chunk_decode_ms,
                "ready_ms": ready_ms,
                "cumulative_ready_ms": cumulative_ready_ms,
                "wav_bytes": chunk_wav_bytes,
            }
        )

    stitched_audio = (
        np.concatenate(stitched_parts).astype(np.float32)
        if stitched_parts
        else np.zeros((0,), dtype=np.float32)
    )
    streaming_summary = {
        "chunk_count": len(chunks),
        "stitched_audio_s": len(stitched_audio) / config.sample_rate,
        "max_chunk_ready_ms": max((chunk["ready_ms"] for chunk in chunks), default=0.0),
        "total_emitted_audio_s": sum(chunk["emitted_audio_s"] for chunk in chunks),
    }
    return chunks, stitched_audio, streaming_summary


def generate_streaming_chunks_sequence(
    head: SNACSequenceHead,
    cond_vector: torch.Tensor,
    snac_code_shapes: list[list[int]],
    decode_codes_to_audio: Callable[[list[torch.Tensor]], tuple[np.ndarray, float]],
    wav_bytes_from_array: Callable[[np.ndarray], bytes],
    config: StreamingConfig,
    perf_counter: Callable[[], float],
) -> tuple[list[dict], np.ndarray, dict]:
    level0_total = int(np.prod(snac_code_shapes[0]))
    windows = level0_stream_windows(level0_total, config)
    chunks = []
    stitched_parts = []
    cumulative_ready_ms = 0.0
    device = cond_vector.device

    for window in windows:
        chunk_started = perf_counter()
        chunk_codes = []
        chunk_level_token_counts = {}
        with torch.inference_mode():
            for level, shape in enumerate(snac_code_shapes):
                level_total = int(np.prod(shape))
                ratio = max(1, level_total // max(level0_total, 1))
                level_start = min(level_total, window["start_level0"] * ratio)
                level_end = min(level_total, window["end_level0"] * ratio)
                required = max(0, level_end - level_start)
                pred = head.generate_level(cond_vector, level, level_end)[0, level_start:level_end]
                chunk_level_token_counts[level] = int(pred.numel())
                stream_shape = list(shape)
                stream_shape[-1] = required
                chunk_codes.append(pred.to(device).view(*stream_shape))
        chunk_head_ms = (perf_counter() - chunk_started) * 1000
        chunk_audio, chunk_decode_ms = decode_codes_to_audio(chunk_codes)
        trim_samples = min(
            len(chunk_audio),
            window["trim_prefix_level0_tokens"] * config.level0_audio_samples,
        )
        emitted_audio = chunk_audio[trim_samples:]
        stitched_parts.append(emitted_audio)
        chunk_wav_bytes = wav_bytes_from_array(chunk_audio)
        ready_ms = chunk_head_ms + chunk_decode_ms
        cumulative_ready_ms += ready_ms
        chunks.append(
            {
                **window,
                "level_token_counts": chunk_level_token_counts,
                "chunk_audio_s": len(chunk_audio) / config.sample_rate,
                "trim_prefix_audio_s": trim_samples / config.sample_rate,
                "emitted_audio_s": len(emitted_audio) / config.sample_rate,
                "head_ms": chunk_head_ms,
                "snac_decode_ms": chunk_decode_ms,
                "ready_ms": ready_ms,
                "cumulative_ready_ms": cumulative_ready_ms,
                "wav_bytes": chunk_wav_bytes,
            }
        )

    stitched_audio = (
        np.concatenate(stitched_parts).astype(np.float32)
        if stitched_parts
        else np.zeros((0,), dtype=np.float32)
    )
    streaming_summary = {
        "chunk_count": len(chunks),
        "stitched_audio_s": len(stitched_audio) / config.sample_rate,
        "max_chunk_ready_ms": max((chunk["ready_ms"] for chunk in chunks), default=0.0),
        "total_emitted_audio_s": sum(chunk["emitted_audio_s"] for chunk in chunks),
    }
    return chunks, stitched_audio, streaming_summary


def generate_streaming_chunks_memory(
    head: SNACMemorySequenceHead,
    memory_vectors: torch.Tensor,
    memory_mask: torch.Tensor,
    snac_code_shapes: list[list[int]],
    decode_codes_to_audio: Callable[[list[torch.Tensor]], tuple[np.ndarray, float]],
    wav_bytes_from_array: Callable[[np.ndarray], bytes],
    config: StreamingConfig,
    perf_counter: Callable[[], float],
) -> tuple[list[dict], np.ndarray, dict]:
    level0_total = int(np.prod(snac_code_shapes[0]))
    windows = level0_stream_windows(level0_total, config)
    chunks = []
    stitched_parts = []
    cumulative_ready_ms = 0.0
    device = memory_vectors.device

    for window in windows:
        chunk_started = perf_counter()
        chunk_codes = []
        chunk_level_token_counts = {}
        with torch.inference_mode():
            for level, shape in enumerate(snac_code_shapes):
                level_total = int(np.prod(shape))
                ratio = max(1, level_total // max(level0_total, 1))
                level_start = min(level_total, window["start_level0"] * ratio)
                level_end = min(level_total, window["end_level0"] * ratio)
                required = max(0, level_end - level_start)
                pred = head.generate_level(memory_vectors, memory_mask, level, level_end)[0, level_start:level_end]
                chunk_level_token_counts[level] = int(pred.numel())
                stream_shape = list(shape)
                stream_shape[-1] = required
                chunk_codes.append(pred.to(device).view(*stream_shape))
        chunk_head_ms = (perf_counter() - chunk_started) * 1000
        chunk_audio, chunk_decode_ms = decode_codes_to_audio(chunk_codes)
        trim_samples = min(
            len(chunk_audio),
            window["trim_prefix_level0_tokens"] * config.level0_audio_samples,
        )
        emitted_audio = chunk_audio[trim_samples:]
        stitched_parts.append(emitted_audio)
        chunk_wav_bytes = wav_bytes_from_array(chunk_audio)
        ready_ms = chunk_head_ms + chunk_decode_ms
        cumulative_ready_ms += ready_ms
        chunks.append(
            {
                **window,
                "level_token_counts": chunk_level_token_counts,
                "chunk_audio_s": len(chunk_audio) / config.sample_rate,
                "trim_prefix_audio_s": trim_samples / config.sample_rate,
                "emitted_audio_s": len(emitted_audio) / config.sample_rate,
                "head_ms": chunk_head_ms,
                "snac_decode_ms": chunk_decode_ms,
                "ready_ms": ready_ms,
                "cumulative_ready_ms": cumulative_ready_ms,
                "wav_bytes": chunk_wav_bytes,
            }
        )

    stitched_audio = (
        np.concatenate(stitched_parts).astype(np.float32)
        if stitched_parts
        else np.zeros((0,), dtype=np.float32)
    )
    streaming_summary = {
        "chunk_count": len(chunks),
        "stitched_audio_s": len(stitched_audio) / config.sample_rate,
        "max_chunk_ready_ms": max((chunk["ready_ms"] for chunk in chunks), default=0.0),
        "total_emitted_audio_s": sum(chunk["emitted_audio_s"] for chunk in chunks),
    }
    return chunks, stitched_audio, streaming_summary


def generate_streaming_chunks_text_scaffold(
    head: SNACTextScaffoldSequenceHead,
    style_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    text_mask: torch.Tensor,
    snac_code_shapes: list[list[int]],
    decode_codes_to_audio: Callable[[list[torch.Tensor]], tuple[np.ndarray, float]],
    wav_bytes_from_array: Callable[[np.ndarray], bytes],
    config: StreamingConfig,
    perf_counter: Callable[[], float],
) -> tuple[list[dict], np.ndarray, dict]:
    level0_total = int(np.prod(snac_code_shapes[0]))
    windows = level0_stream_windows(level0_total, config)
    chunks = []
    stitched_parts = []
    cumulative_ready_ms = 0.0
    device = style_vectors.device

    for window in windows:
        chunk_started = perf_counter()
        chunk_codes = []
        chunk_level_token_counts = {}
        with torch.inference_mode():
            for level, shape in enumerate(snac_code_shapes):
                level_total = int(np.prod(shape))
                ratio = max(1, level_total // max(level0_total, 1))
                level_start = min(level_total, window["start_level0"] * ratio)
                level_end = min(level_total, window["end_level0"] * ratio)
                required = max(0, level_end - level_start)
                pred = head.generate_level(style_vectors, text_vectors, text_mask, level, level_end)[
                    0, level_start:level_end
                ]
                chunk_level_token_counts[level] = int(pred.numel())
                stream_shape = list(shape)
                stream_shape[-1] = required
                chunk_codes.append(pred.to(device).view(*stream_shape))
        chunk_head_ms = (perf_counter() - chunk_started) * 1000
        chunk_audio, chunk_decode_ms = decode_codes_to_audio(chunk_codes)
        trim_samples = min(
            len(chunk_audio),
            window["trim_prefix_level0_tokens"] * config.level0_audio_samples,
        )
        emitted_audio = chunk_audio[trim_samples:]
        stitched_parts.append(emitted_audio)
        chunk_wav_bytes = wav_bytes_from_array(chunk_audio)
        ready_ms = chunk_head_ms + chunk_decode_ms
        cumulative_ready_ms += ready_ms
        chunks.append(
            {
                **window,
                "level_token_counts": chunk_level_token_counts,
                "chunk_audio_s": len(chunk_audio) / config.sample_rate,
                "trim_prefix_audio_s": trim_samples / config.sample_rate,
                "emitted_audio_s": len(emitted_audio) / config.sample_rate,
                "head_ms": chunk_head_ms,
                "snac_decode_ms": chunk_decode_ms,
                "ready_ms": ready_ms,
                "cumulative_ready_ms": cumulative_ready_ms,
                "wav_bytes": chunk_wav_bytes,
            }
        )

    stitched_audio = (
        np.concatenate(stitched_parts).astype(np.float32)
        if stitched_parts
        else np.zeros((0,), dtype=np.float32)
    )
    streaming_summary = {
        "chunk_count": len(chunks),
        "stitched_audio_s": len(stitched_audio) / config.sample_rate,
        "max_chunk_ready_ms": max((chunk["ready_ms"] for chunk in chunks), default=0.0),
        "total_emitted_audio_s": sum(chunk["emitted_audio_s"] for chunk in chunks),
    }
    return chunks, stitched_audio, streaming_summary


def generate_streaming_chunks_text_coarse_to_fine(
    head: SNACTextCoarseToFineSequenceHead,
    style_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    text_mask: torch.Tensor,
    snac_code_shapes: list[list[int]],
    decode_codes_to_audio: Callable[[list[torch.Tensor]], tuple[np.ndarray, float]],
    wav_bytes_from_array: Callable[[np.ndarray], bytes],
    config: StreamingConfig,
    perf_counter: Callable[[], float],
) -> tuple[list[dict], np.ndarray, dict]:
    level0_total = int(np.prod(snac_code_shapes[0]))
    windows = level0_stream_windows(level0_total, config)
    chunks = []
    stitched_parts = []
    cumulative_ready_ms = 0.0
    device = style_vectors.device

    for window in windows:
        chunk_started = perf_counter()
        chunk_codes = []
        chunk_level_token_counts = {}
        generated_prefixes: list[torch.Tensor] = []
        with torch.inference_mode():
            for level, shape in enumerate(snac_code_shapes):
                level_total = int(np.prod(shape))
                ratio = max(1, level_total // max(level0_total, 1))
                level_start = min(level_total, window["start_level0"] * ratio)
                level_end = min(level_total, window["end_level0"] * ratio)
                required = max(0, level_end - level_start)
                coarse_tokens = generated_prefixes[level - 1] if level > 0 else None
                pred_prefix = head.generate_level(
                    style_vectors,
                    text_vectors,
                    text_mask,
                    level,
                    level_end,
                    coarse_tokens=coarse_tokens,
                )
                generated_prefixes.append(pred_prefix)
                pred = pred_prefix[0, level_start:level_end].contiguous()
                chunk_level_token_counts[level] = int(pred.numel())
                stream_shape = list(shape)
                stream_shape[-1] = required
                chunk_codes.append(pred.to(device).view(*stream_shape))
        chunk_head_ms = (perf_counter() - chunk_started) * 1000
        chunk_audio, chunk_decode_ms = decode_codes_to_audio(chunk_codes)
        trim_samples = min(
            len(chunk_audio),
            window["trim_prefix_level0_tokens"] * config.level0_audio_samples,
        )
        emitted_audio = chunk_audio[trim_samples:]
        stitched_parts.append(emitted_audio)
        chunk_wav_bytes = wav_bytes_from_array(chunk_audio)
        ready_ms = chunk_head_ms + chunk_decode_ms
        cumulative_ready_ms += ready_ms
        chunks.append(
            {
                **window,
                "level_token_counts": chunk_level_token_counts,
                "chunk_audio_s": len(chunk_audio) / config.sample_rate,
                "trim_prefix_audio_s": trim_samples / config.sample_rate,
                "emitted_audio_s": len(emitted_audio) / config.sample_rate,
                "head_ms": chunk_head_ms,
                "snac_decode_ms": chunk_decode_ms,
                "ready_ms": ready_ms,
                "cumulative_ready_ms": cumulative_ready_ms,
                "wav_bytes": chunk_wav_bytes,
            }
        )

    stitched_audio = (
        np.concatenate(stitched_parts).astype(np.float32)
        if stitched_parts
        else np.zeros((0,), dtype=np.float32)
    )
    streaming_summary = {
        "chunk_count": len(chunks),
        "stitched_audio_s": len(stitched_audio) / config.sample_rate,
        "max_chunk_ready_ms": max((chunk["ready_ms"] for chunk in chunks), default=0.0),
        "total_emitted_audio_s": sum(chunk["emitted_audio_s"] for chunk in chunks),
    }
    return chunks, stitched_audio, streaming_summary


def generate_streaming_chunks_text_scaffold_projection(
    head: SNACTextScaffoldProjectionHead,
    style_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    text_mask: torch.Tensor,
    snac_code_shapes: list[list[int]],
    decode_codes_to_audio: Callable[[list[torch.Tensor]], tuple[np.ndarray, float]],
    wav_bytes_from_array: Callable[[np.ndarray], bytes],
    config: StreamingConfig,
    perf_counter: Callable[[], float],
) -> tuple[list[dict], np.ndarray, dict]:
    level0_total = int(np.prod(snac_code_shapes[0]))
    windows = level0_stream_windows(level0_total, config)
    chunks = []
    stitched_parts = []
    cumulative_ready_ms = 0.0
    device = style_vectors.device

    for window in windows:
        chunk_started = perf_counter()
        chunk_codes = []
        chunk_level_token_counts = {}
        with torch.inference_mode():
            for level, shape in enumerate(snac_code_shapes):
                level_total = int(np.prod(shape))
                ratio = max(1, level_total // max(level0_total, 1))
                level_start = min(level_total, window["start_level0"] * ratio)
                level_end = min(level_total, window["end_level0"] * ratio)
                required = max(0, level_end - level_start)
                logits = head.forward_level(style_vectors, text_vectors, text_mask, level, length=level_end)
                pred = logits.argmax(dim=-1)[0, level_start:level_end].contiguous()
                chunk_level_token_counts[level] = int(pred.numel())
                stream_shape = list(shape)
                stream_shape[-1] = required
                chunk_codes.append(pred.to(device).view(*stream_shape))
        chunk_head_ms = (perf_counter() - chunk_started) * 1000
        chunk_audio, chunk_decode_ms = decode_codes_to_audio(chunk_codes)
        trim_samples = min(
            len(chunk_audio),
            window["trim_prefix_level0_tokens"] * config.level0_audio_samples,
        )
        emitted_audio = chunk_audio[trim_samples:]
        stitched_parts.append(emitted_audio)
        chunk_wav_bytes = wav_bytes_from_array(chunk_audio)
        ready_ms = chunk_head_ms + chunk_decode_ms
        cumulative_ready_ms += ready_ms
        chunks.append(
            {
                **window,
                "level_token_counts": chunk_level_token_counts,
                "chunk_audio_s": len(chunk_audio) / config.sample_rate,
                "trim_prefix_audio_s": trim_samples / config.sample_rate,
                "emitted_audio_s": len(emitted_audio) / config.sample_rate,
                "head_ms": chunk_head_ms,
                "snac_decode_ms": chunk_decode_ms,
                "ready_ms": ready_ms,
                "cumulative_ready_ms": cumulative_ready_ms,
                "wav_bytes": chunk_wav_bytes,
            }
        )

    stitched_audio = (
        np.concatenate(stitched_parts).astype(np.float32)
        if stitched_parts
        else np.zeros((0,), dtype=np.float32)
    )
    streaming_summary = {
        "chunk_count": len(chunks),
        "stitched_audio_s": len(stitched_audio) / config.sample_rate,
        "max_chunk_ready_ms": max((chunk["ready_ms"] for chunk in chunks), default=0.0),
        "total_emitted_audio_s": sum(chunk["emitted_audio_s"] for chunk in chunks),
    }
    return chunks, stitched_audio, streaming_summary
