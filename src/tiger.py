"""TIGER: RQ-VAE + T5 generative recommendation with constrained beam search."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config

from .t5 import T5Rec
from .rqvae import RQVAE
from .utils import as_batch


class TigerRec(nn.Module):

    @staticmethod
    def _expand_suffix_columns(values: torch.Tensor, codebook_size: int) -> torch.Tensor:
        if values.numel() == 0:
            return values.view(values.shape[0], 0)
        base = max(2, int(codebook_size))
        pieces: list[torch.Tensor] = []
        for col_idx in range(values.shape[1]):
            col = values[:, col_idx].long()
            max_value = int(col.max().item())
            if max_value < base:
                pieces.append(col.unsqueeze(1))
                continue
            digits = max(1, int(math.ceil(math.log(max_value + 1, base))))
            work = col.clone()
            expanded: list[torch.Tensor] = []
            for _ in range(digits):
                expanded.append((work % base).unsqueeze(1))
                work = torch.div(work, base, rounding_mode="floor")
            pieces.extend(expanded)
        if not pieces:
            return torch.zeros((values.shape[0], 0), dtype=torch.long)
        return torch.cat(pieces, dim=1).long()

    def __init__(
        self,
        n_items: int,
        max_seq_length: int = 20,
        embedding_size: int = 64,
        num_codebooks: int = 3,
        codebook_size: int = 256,
        trie_beam_width: int = 30,
        max_new_tokens: int = 10,
        t5_d_model: int = 128,
        t5_d_ff: int = 1024,
        t5_d_kv: int = 64,
        t5_num_layers: int = 4,
        t5_num_decoder_layers: int = 4,
        t5_num_heads: int = 6,
        t5_dropout: float = 0.1,
        t5_temperature: float = 1.0,
        semid_mode: str = "rqvae",
        semid_seed: int = 42,
        semid_kmeans_iters: int = 20,
        semid_rqvae_epochs: int = 50,
        semid_rqvae_batch_size: int = 1024,
        semid_rqvae_lr: float = 1e-3,
        semid_rqvae_weight_decay: float = 1e-5,
        semid_rqvae_hidden_size: int = 256,
        semid_rqvae_code_dim: int = 64,
        semid_rqvae_commitment_weight: float = 0.25,
        semid_rqvae_quant_loss_weight: float = 1.0,
        semid_rqvae_device: str = "cuda",
        data_root: str = "./data",
        dataset_name: str = "",
        item_codes: torch.Tensor | None = None,
        rqvae_tokenizer: RQVAE | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_items = int(n_items)
        self.max_seq_length = int(max_seq_length)
        self.embedding_size = int(embedding_size)
        self.codebook_size = int(codebook_size)
        self.trie_beam_width = max(1, int(trie_beam_width))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.semid_mode = str(semid_mode)
        self.dataset_name = str(dataset_name) if dataset_name else ""

        if item_codes is not None and rqvae_tokenizer is not None:
            self.rqvae_tokenizer = rqvae_tokenizer
            self.rqvae_tokenizer.eval()
            for param in self.rqvae_tokenizer.parameters():
                param.requires_grad = False
            base_books = len(self.rqvae_tokenizer.codebook_size_list)
            self.semid_cache_num_codebooks = int(base_books)
            if int(item_codes.shape[1]) > int(base_books):
                raw_suffix = item_codes[:, base_books:].long()
            else:
                unique_suffix = torch.tensor(self.rqvae_tokenizer.unique_indices, dtype=torch.long)
                raw_suffix = unique_suffix[1:self.n_items + 1].unsqueeze(-1)
            expanded_suffix = self._expand_suffix_columns(raw_suffix, self.codebook_size)
            if expanded_suffix.numel() > 0:
                suffix_table = torch.zeros((self.n_items + 1, expanded_suffix.shape[1]), dtype=torch.long)
                suffix_table[1:] = expanded_suffix
                item_codes = torch.cat([item_codes[:, :base_books], expanded_suffix], dim=1)
            else:
                item_codes = item_codes[:, :base_books]
                suffix_table = torch.zeros((self.n_items + 1, 0), dtype=torch.long)
        else:
            self.rqvae_tokenizer = None
            self.semid_cache_num_codebooks = int(item_codes.shape[1]) if item_codes is not None else int(num_codebooks)
            suffix_table = torch.zeros((self.n_items + 1, 0), dtype=torch.long)

        self.register_buffer("rqvae_suffix_table", suffix_table, persistent=True)
        item_unique_suffix = self.rqvae_suffix_table[:, 0] if int(self.rqvae_suffix_table.shape[1]) == 1 else torch.zeros(self.n_items + 1, dtype=torch.long)
        self.register_buffer("item_unique_suffix", item_unique_suffix, persistent=True)

        if item_codes is not None and int(item_codes.max().item()) >= self.codebook_size:
            raise ValueError(f"Semantic IDs include token index >= codebook_size ({self.codebook_size}).")
        self.num_codebooks = int(item_codes.shape[1]) if item_codes is not None else int(num_codebooks)
        self.register_buffer("item_codes", item_codes if item_codes is not None else torch.zeros(self.n_items + 1, self.num_codebooks, dtype=torch.long), persistent=True)

        token_shift = torch.arange(self.num_codebooks, dtype=torch.long) * self.codebook_size + 2
        self.register_buffer("token_shift", token_shift, persistent=True)
        self.vocab_size = self.codebook_size * self.num_codebooks + 2

        shifted_candidates = (self.item_codes + self.token_shift.unsqueeze(0)).detach().cpu()
        self._all_candidate_tokens: list[list[int]] = shifted_candidates.tolist()
        self._token_to_item: dict[tuple[int, ...], int] = {
            tuple(int(v) for v in row): idx
            for idx, row in enumerate(shifted_candidates.tolist())
        }
        self._code_to_item: dict[tuple[int, ...], int] = {
            tuple(int(v) for v in row): idx
            for idx, row in enumerate(self.item_codes.detach().cpu().tolist())
        }

        config = T5Config(
            vocab_size=self.vocab_size, pad_token_id=0, eos_token_id=1,
            decoder_start_token_id=0, bos_token_id=0,
            d_model=int(t5_d_model), d_ff=int(t5_d_ff), d_kv=int(t5_d_kv),
            num_layers=int(t5_num_layers), num_decoder_layers=int(t5_num_decoder_layers),
            num_heads=int(t5_num_heads), is_encoder_decoder=True,
            relative_attention_num_buckets=32, relative_attention_max_distance=128,
            initializer_factor=1.0, layer_norm_epsilon=1e-6,
            dropout_rate=float(t5_dropout), feed_forward_proj="relu",
            n_positions=max(512, self.max_seq_length * self.num_codebooks + 8),
            use_cache=True,
        )
        self.rec_model = T5Rec(config=config)
        self.rec_model.temperature = float(t5_temperature)

    def _tokenize_items(self, item_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        item_ids = item_ids.long()
        if self.rqvae_tokenizer is not None:
            with torch.no_grad():
                hard_indices, soft_probs, _, _, _, _ = self.rqvae_tokenizer(item_ids)
            hard_indices = hard_indices.long()
            soft_steps = soft_probs.permute(0, 1, 3, 2).float()
            if int(self.rqvae_suffix_table.shape[1]) > 0:
                suffix = self.rqvae_suffix_table[item_ids]
                hard_indices = torch.cat([hard_indices, suffix], dim=-1)
                suffix_probs = F.one_hot(suffix, num_classes=self.codebook_size).float().to(soft_steps.device)
                soft_steps = torch.cat([soft_steps, suffix_probs], dim=2)
            return hard_indices, soft_steps
        raw = (item_ids - 1).clamp(min=0, max=self.n_items - 1)
        hard_indices = self.item_codes[raw]
        soft_steps = F.one_hot(hard_indices, num_classes=self.codebook_size).float()
        return hard_indices, soft_steps

    def _shift_to_token_space(self, hard_indices: torch.Tensor, soft_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, codebook_num = hard_indices.shape
        shifted = hard_indices + self.token_shift.view(1, 1, codebook_num)
        soft = torch.zeros(batch_size, seq_len, codebook_num, self.vocab_size, dtype=soft_steps.dtype, device=soft_steps.device)
        for step in range(codebook_num):
            start = int(self.token_shift[step].item())
            end = start + self.codebook_size
            soft[:, :, step, start:end] = soft_steps[:, :, step, :]
        return shifted.reshape(batch_size, -1), soft.reshape(batch_size, -1, self.vocab_size)

    def _prepare_train_inputs(self, history: torch.Tensor, target: torch.Tensor):
        batch_size = int(history.shape[0])
        device = history.device
        ids = torch.cat([history.long(), target.long().unsqueeze(1)], dim=1)
        hard_indices, soft_steps = self._tokenize_items(ids)
        hard_tokens, soft_embeds = self._shift_to_token_space(hard_indices, soft_steps)
        codebook_num = self.num_codebooks
        input_ids = hard_tokens[:, :-codebook_num].clone()
        inputs_embeds = soft_embeds[:, :-codebook_num, :].clone()
        decoder_target_ids = hard_tokens[:, -codebook_num:].clone()
        decoder_target_embeds = soft_embeds[:, -codebook_num:, :].clone()
        valid_history = history.gt(0).unsqueeze(-1).expand(-1, -1, codebook_num).reshape(batch_size, -1)
        valid_history_emb = valid_history.unsqueeze(-1).expand(-1, -1, self.vocab_size)
        input_ids = input_ids.masked_fill(~valid_history, 0)
        inputs_embeds = inputs_embeds.masked_fill(~valid_history_emb, 0.0)
        input_ids = torch.cat([input_ids, torch.zeros(batch_size, 1, dtype=torch.long, device=device)], dim=1)
        eos_pos = (input_ids > 0).sum(dim=1)
        row_idx = torch.arange(batch_size, device=device)
        input_ids[row_idx, eos_pos] = 1
        inputs_embeds = torch.cat([inputs_embeds, torch.zeros(batch_size, 1, self.vocab_size, dtype=soft_embeds.dtype, device=device)], dim=1)
        labels = torch.cat([decoder_target_ids, torch.ones(batch_size, 1, dtype=torch.long, device=device)], dim=1)
        decoder_input_ids = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device=device), decoder_target_ids], dim=1)
        decoder_inputs_embeds = torch.cat([torch.zeros(batch_size, 1, self.vocab_size, dtype=soft_embeds.dtype, device=device), decoder_target_embeds], dim=1)
        attention_mask = (input_ids != 0).long()
        return input_ids, inputs_embeds, decoder_input_ids, decoder_inputs_embeds, labels, attention_mask

    def _prepare_infer_inputs(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(history.shape[0])
        device = history.device
        hard_indices, soft_steps = self._tokenize_items(history.long())
        hard_tokens, _ = self._shift_to_token_space(hard_indices, soft_steps)
        codebook_num = self.num_codebooks
        valid_history = history.gt(0).unsqueeze(-1).expand(-1, -1, codebook_num).reshape(batch_size, -1)
        input_ids = hard_tokens.masked_fill(~valid_history, 0)
        input_ids = torch.cat([input_ids, torch.zeros(batch_size, 1, dtype=torch.long, device=device)], dim=1)
        eos_pos = (input_ids > 0).sum(dim=1)
        row_idx = torch.arange(batch_size, device=device)
        input_ids[row_idx, eos_pos] = 1
        attention_mask = (input_ids != 0).long()
        return input_ids, attention_mask

    def _decode_token_sequences(self, pred_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, beam_size, seq_len = pred_tokens.shape
        pred_items = torch.full((batch_size, beam_size), -1, dtype=torch.long, device=pred_tokens.device)
        shifts = self.token_shift.detach().cpu().tolist()
        for b in range(batch_size):
            for beam in range(beam_size):
                if seq_len < self.num_codebooks:
                    continue
                decoded: list[int] = []
                valid = True
                for step in range(self.num_codebooks):
                    token = int(pred_tokens[b, beam, step].item())
                    index = token - int(shifts[step])
                    if index < 0 or index >= self.codebook_size:
                        valid = False; break
                    decoded.append(index)
                if not valid:
                    continue
                item = self._code_to_item.get(tuple(decoded))
                if item is not None:
                    pred_items[b, beam] = int(item)
        return pred_items

    def _generate_sequences(self, batch: dict[str, torch.Tensor], beam_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec_batch = as_batch(batch)
        input_ids, attention_mask = self._prepare_infer_inputs(rec_batch.item_id_list)
        return self.rec_model._inference(
            input_ids=input_ids, attention_mask=attention_mask,
            all_indices=self._all_candidate_tokens,
            beam_size=max(1, int(beam_size)),
            max_new_tokens=max(self.max_new_tokens, self.num_codebooks + 2),
            return_sequence_scores=True,
        )

    def calculate_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        rec_batch = as_batch(batch)
        input_ids, inputs_embeds, decoder_input_ids, decoder_inputs_embeds, labels, attention_mask = self._prepare_train_inputs(rec_batch.item_id_list, rec_batch.item_id)
        return self.rec_model._forward(
            input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, labels=labels,
            inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds,
        )

    def calculate_val_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.calculate_loss(batch)

    def full_sort_predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        beam_size = max(self.trie_beam_width, 20)
        pred_tokens, seq_scores = self._generate_sequences(batch, beam_size=beam_size)
        pred_items = self._decode_token_sequences(pred_tokens)
        scores = torch.full((pred_items.shape[0], self.n_items), -1e9, dtype=seq_scores.dtype, device=seq_scores.device)
        for b in range(pred_items.shape[0]):
            for beam in range(pred_items.shape[1]):
                item = int(pred_items[b, beam].item())
                if item < 0:
                    continue
                score = seq_scores[b, beam]
                if score > scores[b, item]:
                    scores[b, item] = score
        empty_rows = scores.eq(-1e9).all(dim=1)
        if bool(empty_rows.any()):
            scores[empty_rows] = 0.0
        return scores

    def generate_topk(self, batch: dict[str, torch.Tensor], k: int = 20) -> torch.Tensor:
        topk = min(max(1, int(k)), self.n_items)
        beam_size = max(self.trie_beam_width, topk)
        pred_tokens, seq_scores = self._generate_sequences(batch, beam_size=beam_size)
        pred_items = self._decode_token_sequences(pred_tokens)
        output = torch.zeros(pred_items.shape[0], topk, dtype=torch.long, device=pred_items.device)
        for b in range(pred_items.shape[0]):
            order = torch.argsort(seq_scores[b], descending=True)
            ranked: list[int] = []
            seen: set[int] = set()
            for idx in order.tolist():
                item = int(pred_items[b, idx].item())
                if item < 0 or item in seen:
                    continue
                ranked.append(item); seen.add(item)
                if len(ranked) >= topk:
                    break
            if not ranked:
                ranked = [0]
            while len(ranked) < topk:
                ranked.append(ranked[-1])
            output[b] = torch.tensor(ranked[:topk], dtype=torch.long, device=output.device)
        return output

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict()}, path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        payload = torch.load(path, map_location=map_location)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        self.load_state_dict(state_dict)
