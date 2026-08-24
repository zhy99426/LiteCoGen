"""T5 encoder-decoder with constrained beam search over semantic ID token space."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack

from .trie import Trie, prefix_allowed_tokens_fn


class ManualT5Stack(T5Stack):

    def __init__(self, config: T5Config, shared_module=None):
        super().__init__(config, shared_module)

    def _forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        cross_attn_head_mask: torch.FloatTensor | None = None,
        past_key_values: tuple | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position=None,
    ):
        if input_ids is None or inputs_embeds is None:
            raise ValueError("ManualT5Stack._forward requires input_ids and inputs_embeds.")
        hard_embeds = self.embed_tokens(input_ids)
        soft_embeds = torch.matmul(inputs_embeds, self.embed_tokens.weight)
        inputs_embeds = (hard_embeds - soft_embeds).detach() + soft_embeds
        return self.forward(
            input_ids=None, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds, head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, cache_position=cache_position,
        )


class T5Rec(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.temperature = 1.0
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = ManualT5Stack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = ManualT5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def _project_logits(self, sequence_output: torch.Tensor) -> torch.Tensor:
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        return self.lm_head(sequence_output)

    def _decode_stack(
        self, decoder_stack, hidden_states, attention_mask,
        decoder_input_ids, decoder_attention_mask,
        past_key_values=None, decoder_inputs_embeds=None,
        decoder_head_mask=None, cross_attn_head_mask=None,
        use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None,
    ):
        if decoder_inputs_embeds is not None:
            decoder_outputs = decoder_stack._forward(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values, use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            decoder_outputs = decoder_stack.forward(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        sequence_output = decoder_outputs[0]
        lm_logits = self._project_logits(sequence_output)
        return decoder_outputs, sequence_output, lm_logits

    def _forward(
        self,
        input_ids=None, attention_mask=None, encoder_outputs=None,
        decoder_input_ids=None, decoder_attention_mask=None,
        cross_attn_head_mask=None, past_key_values=None,
        use_cache=None, labels=None,
        inputs_embeds=None, decoder_inputs_embeds=None,
        head_mask=None, decoder_head_mask=None,
        output_attentions=None, output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask
        if encoder_outputs is None:
            if inputs_embeds is not None:
                encoder_outputs = self.encoder._forward(
                    input_ids=input_ids, attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds, head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                encoder_outputs = self.encoder.forward(
                    input_ids=input_ids, attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds, head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)
        decoder_outputs, _sequence_output, lm_logits = self._decode_stack(
            decoder_stack=self.decoder, hidden_states=hidden_states,
            attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        loss = (
            CrossEntropyLoss(ignore_index=-100, reduction="mean")(
                (lm_logits / self.temperature).view(-1, lm_logits.size(-1)),
                labels.view(-1).to(lm_logits.device),
            )
            if labels is not None else None
        )
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return (loss,) + output if loss is not None else output
        return loss

    def _inference(
        self, input_ids, attention_mask, all_indices,
        beam_size, max_new_tokens=10,
        return_sequence_scores=False,
        encoder_outputs=None, logits_processor=None,
    ):
        if not isinstance(all_indices, list):
            all_indices = all_indices.tolist()
        if len(all_indices) == 0:
            fb = max(1, int(beam_size))
            pred = torch.zeros(input_ids.size(0), fb, 0, dtype=torch.long, device=input_ids.device)
            if not return_sequence_scores:
                return pred
            return pred, torch.zeros(input_ids.size(0), fb, dtype=torch.float32, device=input_ids.device)
        beam_size = min(max(1, int(beam_size)), len(all_indices))
        max_candidate_len = max(len(c) for c in all_indices)
        max_new_tokens = max(1, min(int(max_new_tokens), int(max_candidate_len) + 2))
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        candidate_trie = Trie([[0] + c + [1] for c in all_indices])
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
        generate_kwargs = dict(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=beam_size, num_return_sequences=beam_size,
            output_scores=True, return_dict_in_generate=True,
            early_stopping=True, encoder_outputs=encoder_outputs,
        )
        if logits_processor is not None:
            generate_kwargs["logits_processor"] = [logits_processor]
        output = self.generate(**generate_kwargs)
        batch_size = input_ids.size(0)
        sequences = output["sequences"]
        pred = torch.stack([seq[1:-1] for seq in sequences]).view(batch_size, beam_size, -1)
        if not return_sequence_scores:
            return pred
        seq_scores = output.get("sequences_scores", None)
        if seq_scores is None:
            seq_scores = torch.zeros(batch_size * beam_size, dtype=torch.float32, device=pred.device)
        return pred, seq_scores.view(batch_size, beam_size)
