"""Subject-model loading and residual-stream hooks for activation extraction.

The extraction scripts tokenize and pool themselves; this module loads the subject model,
resolves the hooked block, captures its output, and locates sentence spans in the text.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from nld.core.types import BehaviorExample


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


@dataclass
class HookSpec:
    """Where to read activations. Only stream='resid_post' (the output of decoder block
    `layer`) is supported. token_selector and normalize are recorded for reference; the
    extraction scripts do their own pooling and do not normalize."""
    layer: int
    stream: str = "resid_post"
    token_selector: str = "last_token"
    normalize: bool = False


class ActivationExtractor:
    def __init__(self, model_spec: ModelSpec, hook_spec: HookSpec):
        self.model_spec = model_spec
        self.hook_spec = hook_spec
        self.tokenizer = None
        self.model = None
        self.device = torch.device(model_spec.device if torch.cuda.is_available() else "cpu")
        self._load_model_and_tokenizer()

    # -------------------------
    # Model loading
    # -------------------------

    def _load_model_and_tokenizer(self) -> None:
        if getattr(self, "model", None) is not None:             # idempotent: already loaded
            return
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

        # load shards directly onto the GPU when one is available; otherwise load on CPU
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

    # -------------------------
    # Text formatting
    # -------------------------

    def _format_example_text(self, ex: BehaviorExample) -> dict[str, Any]:
        """The text run through the subject model: prompt and response joined by a newline,
        or the prompt alone when there is no response."""
        if ex.response is not None:
            return {"text": f"{ex.prompt}\n{ex.response}", "example_id": ex.example_id}
        return {"text": ex.prompt, "example_id": ex.example_id}

    # -------------------------
    # Hooks
    # -------------------------

    def _resolve_hook_module(self) -> nn.Module:
        """The decoder block whose output is the residual stream after `layer`."""
        blocks = self._get_transformer_blocks()
        layer_idx = self.hook_spec.layer
        if layer_idx < 0 or layer_idx >= len(blocks):
            raise ValueError(f"Requested layer {layer_idx}, but model only has {len(blocks)} blocks.")
        if self.hook_spec.stream != "resid_post":
            raise ValueError(f"Unsupported stream: {self.hook_spec.stream}")
        return blocks[layer_idx]

    def _get_transformer_blocks(self):
        # search the model and its common wrapper attributes for the decoder stack
        candidates = [self.model]
        for attr in ("model", "base_model", "language_model"):
            candidates.extend([getattr(obj, attr) for obj in candidates if hasattr(obj, attr)])

        seen_ids: set[int] = set()
        unique_candidates = []
        for obj in candidates:
            if id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                unique_candidates.append(obj)

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
            f"Model class: {self.model.__class__.__name__}."
        )

    def _make_hook_fn(self, captured: dict[str, torch.Tensor]) -> Callable:
        """Forward hook that stores the block's hidden states [B, T, D] in captured['hidden']."""
        def hook_fn(module: nn.Module, inputs: tuple, output: Any):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError(f"Hook output was not a tensor. Got type: {type(hidden)}")
            captured["hidden"] = hidden.detach()

        return hook_fn

    # -------------------------
    # Sentence spans
    # -------------------------

    @staticmethod
    def _split_sentences(text: str, min_chars: int = 10) -> list[str]:
        """Split into sentences on terminal punctuation; drop very short fragments."""
        parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        return [p.strip() for p in parts if len(p.strip()) >= min_chars]

    def _find_char_span(self, full_text: str, sentence_text: str) -> tuple[int, int] | None:
        """Character span of sentence_text in full_text, matching exactly or else up to
        whitespace differences."""
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

    def _normalize_span_text(self, text: str) -> str:
        return " ".join(text.split())

    def _recover_span_from_normalized(
        self,
        original_text: str,
        normalized_substring: str,
        normalized_start: int,
    ) -> tuple[int, int] | None:
        """Map a span in the whitespace-normalized text back to the original text."""
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
