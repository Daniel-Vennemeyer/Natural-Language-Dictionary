from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from taxonomy_discovery.core.types import ActivationBatch, BehaviorExample


@dataclass
class ModelSpec:
    model_name: str
    device: str = "cuda"
    dtype: str = "auto"  # auto | float16 | bfloat16 | float32
    trust_remote_code: bool = False
    local_files_only: bool = False
    revision: str | None = None
    batch_size: int = 8
    max_length: int = 2048
    hidden_size: int | None = None


@dataclass
class HookSpec:
    layer: int
    stream: str = "resid_post"
    token_selector: str = "last_token"
    normalize: bool = False
    token_selector_kwargs: dict[str, Any] = field(default_factory=dict)


class ActivationExtractor:
    """
    Realistic v1 activation extractor for HF causal LMs.

    Supported stream values:
      - resid_post: output of transformer block
      - mlp_out: if available, hook MLP module output
      - attn_out: if available, hook attention module output

    Supported token selectors:
      - last_token
      - mean_all_tokens
      - last_non_padding_token
      - assistant_last_token  (placeholder fallback to last_non_padding_token)
      - response_eos          (last EOS token, with fallback to last non-padding)
      - sentence_span_mean    (mean-pool a labeled sentence span inside the response)
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        hook_spec: HookSpec,
    ):
        self.model_spec = model_spec
        self.hook_spec = hook_spec

        self.tokenizer = None
        self.model = None
        self.device = torch.device(model_spec.device if torch.cuda.is_available() else "cpu")

        self._load_model_and_tokenizer()

    # -------------------------
    # Public API
    # -------------------------

    def extract(self, examples: list[BehaviorExample]) -> ActivationBatch:
        if len(examples) == 0:
            raise ValueError("No examples provided to ActivationExtractor.extract().")

        all_vectors: list[np.ndarray] = []
        all_ids: list[str] = []

        total_batches = math.ceil(len(examples) / self.model_spec.batch_size)
        batch_gen = self._batched(examples, self.model_spec.batch_size)
        for batch_examples in tqdm(batch_gen, desc="Processing batches", unit="batch", total=total_batches):
            batch_vectors = self._extract_batch(batch_examples)
            all_vectors.append(batch_vectors)
            all_ids.extend([ex.example_id for ex in batch_examples])

        activations = np.concatenate(all_vectors, axis=0).astype(np.float32)

        if self.hook_spec.normalize:
            norms = np.linalg.norm(activations, axis=1, keepdims=True) + 1e-8
            activations = activations / norms

        return ActivationBatch(
            example_ids=all_ids,
            activations=activations,
            metadata={
                "model_name": self.model_spec.model_name,
                "device": str(self.device),
                "layer": self.hook_spec.layer,
                "stream": self.hook_spec.stream,
                "token_selector": self.hook_spec.token_selector,
                "normalize": self.hook_spec.normalize,
                "n_examples": len(examples),
                "hidden_size": int(activations.shape[1]),
            },
        )

    def extract_per_token(self, examples: list[BehaviorExample]) -> ActivationBatch:
        """Per-token activations: every non-padding token becomes a row.

        Returns an ActivationBatch with activations [N_tokens, d] and example_ids
        REPEATED per token (so each token carries its source example's id). Optionally
        caps tokens/example via hook_spec.token_selector_kwargs['max_tokens_per_example']
        (evenly subsampled) to bound size. This is the natural SAE input and gives ~100x
        more training points than one pooled vector per example.
        """
        if len(examples) == 0:
            raise ValueError("No examples provided to extract_per_token().")
        cap = int(self.hook_spec.token_selector_kwargs.get("max_tokens_per_example", 0))
        rng = np.random.default_rng(0)

        all_vecs: list[np.ndarray] = []
        all_ids: list[str] = []
        all_pos: list[int] = []
        total_batches = math.ceil(len(examples) / self.model_spec.batch_size)
        for batch in tqdm(self._batched(examples, self.model_spec.batch_size),
                          desc="Per-token batches", unit="batch", total=total_batches):
            texts = [self._format_example_text(ex)["text"] for ex in batch]
            tok = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.model_spec.max_length)
            tok = {k: v.to(self.device) for k, v in tok.items()}
            captured: dict[str, torch.Tensor] = {}
            handle = self._resolve_hook_module().register_forward_hook(self._make_hook_fn(captured))
            try:
                with torch.no_grad():
                    self.model(**tok)
            finally:
                handle.remove()
            hidden = captured["hidden"]                      # [B, T, D]
            mask = tok["attention_mask"]                     # [B, T]
            for i, ex in enumerate(batch):
                vecs, pos = self._flatten_valid(hidden[i], mask[i], cap, rng)
                all_vecs.append(vecs.detach().to(torch.float32).cpu().numpy())
                all_ids.extend([ex.example_id] * vecs.shape[0])
                all_pos.extend(pos)

        activations = np.concatenate(all_vecs, axis=0).astype(np.float32)
        if self.hook_spec.normalize:
            activations = activations / (np.linalg.norm(activations, axis=1, keepdims=True) + 1e-8)
        return ActivationBatch(
            example_ids=all_ids,
            activations=activations,
            metadata={
                "model_name": self.model_spec.model_name, "device": str(self.device),
                "layer": self.hook_spec.layer, "stream": self.hook_spec.stream,
                "token_selector": "all_tokens", "normalize": self.hook_spec.normalize,
                "per_token": True, "n_examples": len(examples),
                "n_tokens": int(activations.shape[0]), "token_positions": all_pos,
                "hidden_size": int(activations.shape[1]),
                "max_tokens_per_example": cap,
            },
        )

    def extract_per_sentence(self, examples: list[BehaviorExample]) -> ActivationBatch:
        """Per-sentence activations: each response sentence becomes one row (mean-pooled
        over its token span). Carries the sentence text per row in metadata['row_text'] so
        downstream presence/score supervision is SENTENCE-LOCAL (no per-token dilution, no
        whole-response averaging). Uses offset-mapping so token spans align with the padded
        hidden positions for either padding side.
        """
        if len(examples) == 0:
            raise ValueError("No examples provided to extract_per_sentence().")
        cap = int(self.hook_spec.token_selector_kwargs.get("max_sentences_per_example", 0))

        all_vecs: list[np.ndarray] = []
        all_ids: list[str] = []
        all_texts: list[str] = []
        total_batches = math.ceil(len(examples) / self.model_spec.batch_size)
        for batch in tqdm(self._batched(examples, self.model_spec.batch_size),
                          desc="Per-sentence batches", unit="batch", total=total_batches):
            texts = [self._format_example_text(ex)["text"] for ex in batch]
            tok = self.tokenizer(texts, return_tensors="pt", return_offsets_mapping=True,
                                 padding=True, truncation=True, max_length=self.model_spec.max_length)
            offsets = tok.pop("offset_mapping")              # [B, T, 2]
            tok = {k: v.to(self.device) for k, v in tok.items()}
            captured: dict[str, torch.Tensor] = {}
            handle = self._resolve_hook_module().register_forward_hook(self._make_hook_fn(captured))
            try:
                with torch.no_grad():
                    self.model(**tok)
            finally:
                handle.remove()
            hidden = captured["hidden"]                      # [B, T, D]
            for i, ex in enumerate(batch):
                full_text = texts[i]
                off_i = offsets[i].tolist()
                sents = self._split_sentences(ex.response or "")
                if cap and len(sents) > cap:
                    idx = np.linspace(0, len(sents) - 1, cap).round().astype(int)
                    sents = [sents[j] for j in sorted(set(idx.tolist()))]
                for sent in sents:
                    span = self._find_char_span(full_text, sent)
                    if span is None:
                        continue
                    cs, ce = span
                    pos = [t for t, (s, e) in enumerate(off_i) if e > s and s < ce and e > cs]
                    if not pos:
                        continue
                    vec = hidden[i, pos].mean(dim=0)
                    all_vecs.append(vec.detach().to(torch.float32).cpu().numpy())
                    all_ids.append(ex.example_id)
                    all_texts.append(sent)

        activations = np.stack(all_vecs, axis=0).astype(np.float32)
        if self.hook_spec.normalize:
            activations = activations / (np.linalg.norm(activations, axis=1, keepdims=True) + 1e-8)
        return ActivationBatch(
            example_ids=all_ids,
            activations=activations,
            metadata={
                "model_name": self.model_spec.model_name, "device": str(self.device),
                "layer": self.hook_spec.layer, "stream": self.hook_spec.stream,
                "token_selector": "all_sentences", "normalize": self.hook_spec.normalize,
                "per_sentence": True, "per_token": False, "row_text": all_texts,
                "n_examples": len(examples), "n_sentences": int(activations.shape[0]),
                "hidden_size": int(activations.shape[1]),
                "max_sentences_per_example": cap,
            },
        )

    @staticmethod
    def _split_sentences(text: str, min_chars: int = 10) -> list[str]:
        """Split into sentences on terminal punctuation; drop very short fragments."""
        parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        return [p.strip() for p in parts if len(p.strip()) >= min_chars]

    @staticmethod
    def _flatten_valid(hidden_row: torch.Tensor, mask_row: torch.Tensor, cap: int, rng):
        """Non-padding token vectors for one example, optionally capped (even subsample)."""
        valid = mask_row.to(torch.bool)
        idx = valid.nonzero(as_tuple=False).squeeze(-1)          # positions of real tokens
        if cap and cap > 0 and idx.numel() > cap:
            sel = np.linspace(0, idx.numel() - 1, cap).round().astype(int)
            idx = idx[torch.as_tensor(sel, device=idx.device)]
        return hidden_row[idx], idx.detach().cpu().tolist()

    # -------------------------
    # Model loading
    # -------------------------

    def _load_model_and_tokenizer(self) -> None:
        if getattr(self, "model", None) is not None:             # idempotent: __init__ already
            return                                               # loaded; a 2nd copy OOMs at 35B
        resolved_dtype = self._resolve_dtype(self.model_spec.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_spec.model_name,
            trust_remote_code=self.model_spec.trust_remote_code,
            local_files_only=self.model_spec.local_files_only,
            revision=self.model_spec.revision,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        # Stream shards straight to the accelerator (skip the CPU staging copy).
        # On NFS-backed $HOME this ~halves load time for a 30B MoE vs
        # `.from_pretrained(...).to(device)`. Falls back to CPU load when
        # CUDA is unavailable.
        device_str = str(self.device)
        use_accelerate = device_str.startswith("cuda") and torch.cuda.is_available()
        device_map = device_str if use_accelerate else None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_spec.model_name,
            trust_remote_code=self.model_spec.trust_remote_code,
            local_files_only=self.model_spec.local_files_only,
            revision=self.model_spec.revision,
            dtype=resolved_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        if not use_accelerate:
            self.model.to(self.device)

    def _resolve_dtype(self, dtype_str: str):
        if dtype_str == "auto":
            return "auto"
        if dtype_str == "float16":
            return torch.float16
        if dtype_str == "bfloat16":
            return torch.bfloat16
        if dtype_str == "float32":
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    def unload(self) -> None:
        try:
            if getattr(self, "model", None) is not None:
                try:
                    self.model = self.model.to("cpu")
                except Exception:
                    pass
                del self.model
                self.model = None
        except Exception:
            self.model = None

        try:
            if getattr(self, "tokenizer", None) is not None:
                del self.tokenizer
                self.tokenizer = None
        except Exception:
            self.tokenizer = None

    # -------------------------
    # Batch extraction
    # -------------------------

    def _extract_batch(self, examples: list[BehaviorExample]) -> np.ndarray:
        formatted_examples = [self._format_example_text(ex) for ex in examples]
        texts = [item["text"] for item in formatted_examples]

        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_spec.max_length,
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        captured: dict[str, torch.Tensor] = {}

        hook_module = self._resolve_hook_module()
        hook_fn = self._make_hook_fn(captured)

        handle = hook_module.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                _ = self.model(**tokenized)
        finally:
            handle.remove()

        if "hidden" not in captured:
            raise RuntimeError("Hook did not capture activations. Check layer/stream resolution.")

        hidden = captured["hidden"]
        # Expected shape: [batch, seq, hidden]
        if hidden.ndim != 3:
            raise RuntimeError(f"Expected 3D hidden tensor, got shape {tuple(hidden.shape)}")

        selected = self._select_token_vectors(
            hidden_states=hidden,
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            examples=examples,
            texts=texts,
            formatted_examples=formatted_examples,
        )
        return selected.detach().to(torch.float32).cpu().numpy()

    def _format_example_text(self, ex: BehaviorExample) -> dict[str, Any]:
        """
        Returns a dict rather than a raw string so token selection can use
        auxiliary formatting metadata (for example, sentence-span pooling).
        """
        metadata = dict(ex.metadata or {})

        if metadata.get("annotation_level") == "sentence" and metadata.get("sentence_text") and ex.response is not None:
            return {
                "text": ex.response,
                "sentence_text": str(metadata["sentence_text"]),
                "annotation_level": "sentence",
                "example_id": ex.example_id,
            }

        if ex.response is not None:
            return {
                "text": f"{ex.prompt}\n{ex.response}",
                "annotation_level": "response",
                "example_id": ex.example_id,
            }

        return {
            "text": ex.prompt,
            "annotation_level": "prompt",
            "example_id": ex.example_id,
        }

    # -------------------------
    # Hook resolution
    # -------------------------

    def _resolve_hook_module(self) -> nn.Module:
        """
        Tries to resolve a model submodule for the configured layer/stream.
        This is intentionally conservative and aimed at common HF decoder architectures.

        resid_post:
            hook the transformer block itself and use its output
        mlp_out:
            hook block.mlp
        attn_out:
            hook block.self_attn or block.attn
        """
        blocks = self._get_transformer_blocks()
        layer_idx = self.hook_spec.layer

        if layer_idx < 0 or layer_idx >= len(blocks):
            raise ValueError(
                f"Requested layer {layer_idx}, but model only has {len(blocks)} blocks."
            )

        block = blocks[layer_idx]

        if self.hook_spec.stream == "resid_post":
            return block

        if self.hook_spec.stream == "mlp_out":
            if hasattr(block, "mlp"):
                return block.mlp
            raise ValueError("Block has no .mlp module for stream='mlp_out'.")

        if self.hook_spec.stream == "attn_out":
            if hasattr(block, "self_attn"):
                return block.self_attn
            if hasattr(block, "attn"):
                return block.attn
            raise ValueError("Block has no attention module for stream='attn_out'.")

        raise ValueError(f"Unsupported stream: {self.hook_spec.stream}")

    def _get_transformer_blocks(self):
        model = self.model

        # Unwrap common wrapper attributes first.
        candidates = [model]
        for attr in ("model", "base_model", "language_model"):
            next_candidates = []
            for obj in candidates:
                if hasattr(obj, attr):
                    next_candidates.append(getattr(obj, attr))
            candidates.extend(next_candidates)

        seen_ids: set[int] = set()
        unique_candidates = []
        for obj in candidates:
            obj_id = id(obj)
            if obj_id not in seen_ids:
                seen_ids.add(obj_id)
                unique_candidates.append(obj)

        # Common decoder stacks:
        # - Llama / Qwen / Mistral / Gemma: *.layers
        # - GPT2-like: *.h or *.transformer.h
        for obj in unique_candidates:
            if hasattr(obj, "layers"):
                return obj.layers
            if hasattr(obj, "h"):
                return obj.h
            if hasattr(obj, "transformer") and hasattr(obj.transformer, "h"):
                return obj.transformer.h
            if hasattr(obj, "decoder") and hasattr(obj.decoder, "layers"):
                return obj.decoder.layers


        raise ValueError(
            "Could not find transformer blocks on model. "
            f"Model class: {self.model.__class__.__name__}. "
            "Add architecture-specific support in _get_transformer_blocks()."
        )

    def _make_hook_fn(self, captured: dict[str, torch.Tensor]) -> Callable:
        def hook_fn(module: nn.Module, inputs: tuple, output: Any):
            hidden = output

            # Many modules return tuples, especially attention modules.
            if isinstance(hidden, tuple):
                hidden = hidden[0]

            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError(
                    f"Hook output was not a tensor. Got type: {type(hidden)}"
                )

            captured["hidden"] = hidden.detach()

        return hook_fn

    # -------------------------
    # Token selection
    # -------------------------

    def _select_token_vectors(
        self,
        hidden_states: torch.Tensor,   # [B, T, D]
        input_ids: torch.Tensor,       # [B, T]
        attention_mask: torch.Tensor,  # [B, T]
        examples: list[BehaviorExample],
        texts: list[str],
        formatted_examples: list[dict[str, Any]],
    ) -> torch.Tensor:
        selector = self.hook_spec.token_selector

        if selector == "last_token":
            return hidden_states[:, -1, :]

        if selector == "mean_all_tokens":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return summed / denom

        if selector == "last_non_padding_token":
            return self._select_last_non_padding_token(hidden_states, attention_mask)

        if selector == "assistant_last_token":
            # v1 fallback:
            # later you can replace this with chat-template aware selection.
            return self._select_last_non_padding_token(hidden_states, attention_mask)

        if selector == "response_eos":
            return self._select_response_eos_token(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if selector == "sentence_span_mean":
            return self._select_sentence_span_mean(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                examples=examples,
                texts=texts,
                formatted_examples=formatted_examples,
            )

        raise ValueError(f"Unsupported token selector: {selector}")

    def _select_last_non_padding_token(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Index of the LAST valid (non-padding) token per row, correct for both
        # left- and right-padding. The previous `sum(mask) - 1` formula assumed
        # right padding; with the tokenizer's left padding it indexed into the
        # padding region for any sequence shorter than the batch max, returning
        # padding-token activations. (mask * position).argmax gives the largest
        # position with mask == 1, i.e. the true last real token in either layout.
        seq_len = attention_mask.shape[1]
        positions = torch.arange(seq_len, device=hidden_states.device)
        last_idx = (attention_mask * positions).argmax(dim=1)
        batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_idx, last_idx, :]


    def _select_response_eos_token(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            return self._select_last_non_padding_token(hidden_states, attention_mask)

        selected: list[torch.Tensor] = []
        for row_hidden, row_ids, row_mask in zip(hidden_states, input_ids, attention_mask):
            valid = row_mask.to(torch.bool)
            eos_positions = ((row_ids == eos_token_id) & valid).nonzero(as_tuple=False).squeeze(-1)
            if eos_positions.numel() == 0:
                selected.append(
                    self._select_last_non_padding_token(
                        row_hidden.unsqueeze(0),
                        row_mask.unsqueeze(0),
                    )[0]
                )
                continue
            selected.append(row_hidden[int(eos_positions[-1].item())])

        return torch.stack(selected, dim=0)


    def _select_sentence_span_mean(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        examples: list[BehaviorExample],
        texts: list[str],
        formatted_examples: list[dict[str, Any]],
    ) -> torch.Tensor:
        selected: list[torch.Tensor] = []

        for i, ex in enumerate(examples):
            formatted = formatted_examples[i]
            sentence_text = str(formatted.get("sentence_text", "")).strip()
            full_text = texts[i]

            if not sentence_text:
                selected.append(self._select_last_non_padding_token(hidden_states[i:i+1], attention_mask[i:i+1])[0])
                continue

            span = self._find_char_span(full_text, sentence_text)
            if span is None:
                selected.append(self._select_last_non_padding_token(hidden_states[i:i+1], attention_mask[i:i+1])[0])
                continue

            token_span = self._char_span_to_token_span(
                text=full_text,
                char_span=span,
            )
            if token_span is None:
                selected.append(self._select_last_non_padding_token(hidden_states[i:i+1], attention_mask[i:i+1])[0])
                continue

            token_start, token_end = token_span
            row_hidden = hidden_states[i]
            row_mask = attention_mask[i]

            valid_positions = row_mask.nonzero(as_tuple=False).squeeze(-1)
            if valid_positions.numel() == 0:
                selected.append(row_hidden[-1])
                continue

            seq_len = int(valid_positions[-1].item()) + 1
            token_start = max(0, min(int(token_start), seq_len - 1))
            token_end = max(token_start, min(int(token_end), seq_len))

            span_hidden = row_hidden[token_start:token_end]
            if span_hidden.numel() == 0:
                selected.append(self._select_last_non_padding_token(hidden_states[i:i+1], attention_mask[i:i+1])[0])
                continue

            selected.append(span_hidden.mean(dim=0))

        return torch.stack(selected, dim=0)

    def _find_char_span(self, full_text: str, sentence_text: str) -> tuple[int, int] | None:
        direct_idx = full_text.find(sentence_text)
        if direct_idx >= 0:
            return (direct_idx, direct_idx + len(sentence_text))

        normalized_full = self._normalize_span_text(full_text)
        normalized_sentence = self._normalize_span_text(sentence_text)
        norm_idx = normalized_full.find(normalized_sentence)
        if norm_idx >= 0:
            recovered = self._recover_span_from_normalized(full_text, normalized_sentence, norm_idx)
            if recovered is not None:
                return recovered

        return None

    def _char_span_to_token_span(
        self,
        text: str,
        char_span: tuple[int, int],
    ) -> tuple[int, int] | None:
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.model_spec.max_length,
            add_special_tokens=True,
        )
        offsets = encoding.get("offset_mapping")
        if offsets is None:
            return None

        char_start, char_end = char_span
        token_start = None
        token_end = None

        for idx, (start, end) in enumerate(offsets):
            if start == end:
                continue
            if token_start is None and start <= char_start < end:
                token_start = idx
            if start < char_end <= end:
                token_end = idx + 1
                break
            if token_start is not None and start >= char_end:
                token_end = idx
                break

        if token_start is None:
            for idx, (start, end) in enumerate(offsets):
                if start == end:
                    continue
                if start >= char_start:
                    token_start = idx
                    break

        if token_end is None and token_start is not None:
            for idx in range(token_start, len(offsets)):
                start, end = offsets[idx]
                if start == end:
                    continue
                if end >= char_end:
                    token_end = idx + 1
                    break
            if token_end is None:
                token_end = len(offsets)

        if token_start is None or token_end is None or token_end <= token_start:
            return None
        return (token_start, token_end)

    def _normalize_span_text(self, text: str) -> str:
        return " ".join(text.split())

    def _recover_span_from_normalized(
        self,
        original_text: str,
        normalized_substring: str,
        normalized_start: int,
    ) -> tuple[int, int] | None:
        normalized_chars: list[str] = []
        normalized_to_original: list[int] = []

        in_space = False
        for idx, ch in enumerate(original_text):
            if ch.isspace():
                if not in_space and normalized_chars:
                    normalized_chars.append(" ")
                    normalized_to_original.append(idx)
                in_space = True
            else:
                normalized_chars.append(ch)
                normalized_to_original.append(idx)
                in_space = False

        if normalized_chars and normalized_chars[-1] == " ":
            normalized_chars.pop()
            normalized_to_original.pop()

        rebuilt = "".join(normalized_chars)
        if normalized_start < 0 or normalized_start + len(normalized_substring) > len(rebuilt):
            return None
        if rebuilt[normalized_start:normalized_start + len(normalized_substring)] != normalized_substring:
            return None

        orig_start = normalized_to_original[normalized_start]
        orig_end = normalized_to_original[normalized_start + len(normalized_substring) - 1] + 1
        return (orig_start, orig_end)

    # -------------------------
    # Utilities
    # -------------------------

    @staticmethod
    def _batched(items: list[Any], batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
