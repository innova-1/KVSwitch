"""
KV-Cache operations — low-level DynamicCache manipulation and variant pre-computation.

Extracted and generalized from method2_kvcache.py. Supports:
  - API-version-agnostic cache access (transformers 4.45+ and legacy)
  - Prefix splicing (splice_prefix)
  - Token range removal (remove_token_range) — NEW for Phase 3 summary replacement
  - Generation helpers (prefill, sample_token, generate_one_token, generate_until_stop)
  - KVVariantBuilder: generalized 2^N variant pre-computation for arbitrary field sets
"""

from __future__ import annotations

import re
import torch
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional, Callable

from transformers.cache_utils import DynamicCache

from core.sensitive_config import SensitiveRegistry
from core.prompt_builder import PromptBuilder

# Regex to strip <think> blocks (model-specific)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks before parsing control tokens."""
    if not text:
        return text
    return _THINK_PATTERN.sub("", text)


# ═══════════════════════════════════════════════════════════════════════════
# DynamicCache accessors (API-version agnostic)
# ═══════════════════════════════════════════════════════════════════════════


def get_cache_num_layers(cache: DynamicCache) -> int:
    """Return number of layers.

    Handles new API  (4.45+: cache.layers[i].keys/.values)
    and legacy API (cache.key_cache[i] / cache.value_cache[i]).
    """
    if cache is None:
        return 0
    if hasattr(cache, "layers") and cache.layers is not None:
        return len(cache.layers)
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    return 0


def get_cache_kv(cache: DynamicCache, layer_idx: int):
    """Get (key_tensor, value_tensor) for layer `layer_idx`."""
    if hasattr(cache, "layers") and cache.layers is not None:
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def set_cache_kv(cache: DynamicCache, layer_idx: int,
                 key: torch.Tensor, value: torch.Tensor) -> None:
    """Set (key_tensor, value_tensor) for layer `layer_idx`."""
    if hasattr(cache, "layers") and cache.layers is not None:
        cache.layers[layer_idx].keys = key
        cache.layers[layer_idx].values = value
    else:
        cache.key_cache[layer_idx] = key
        cache.value_cache[layer_idx] = value


def get_cache_seq_len(cache: DynamicCache) -> int:
    """Get current sequence length from the first layer's key tensor."""
    k, _ = get_cache_kv(cache, 0)
    return k.shape[2]


# ═══════════════════════════════════════════════════════════════════════════
# Cache manipulation
# ═══════════════════════════════════════════════════════════════════════════


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """Deep-copy a DynamicCache (tensors detached + cloned).

    Returns None if cache is None.
    """
    if cache is None:
        return None
    new = DynamicCache()
    n = get_cache_num_layers(cache)
    for i in range(n):
        k, v = get_cache_kv(cache, i)
        new.update(k.detach().clone(), v.detach().clone(), i)
    return new


def splice_prefix(
    source: DynamicCache,
    target: DynamicCache,
    prefix_len: int,
) -> DynamicCache:
    """Replace the first `prefix_len` positions of target with source's prefix.

    Only positions [0, prefix_len) are swapped.
    Positions [prefix_len, ...) in target stay unchanged.

    Returns a NEW DynamicCache (does not mutate inputs).
    """
    out = clone_cache(target)
    n = get_cache_num_layers(out)
    for i in range(n):
        src_k, src_v = get_cache_kv(source, i)
        tgt_k, tgt_v = get_cache_kv(out, i)
        new_k = torch.cat(
            [src_k[:, :, :prefix_len, :], tgt_k[:, :, prefix_len:, :]], dim=2
        )
        new_v = torch.cat(
            [src_v[:, :, :prefix_len, :], tgt_v[:, :, prefix_len:, :]], dim=2
        )
        set_cache_kv(out, i, new_k, new_v)
    return out


def remove_token_range(
    cache: DynamicCache,
    start: int,
    end: int,
) -> DynamicCache:
    """Remove token positions [start, end) from ALL layers of the cache.

    Concatenates [0:start) + [end:total) for each layer's key and value.
    Returns a NEW DynamicCache.

    Used in Method 2 Phase 3 to surgically remove sensitive output tokens
    from the KV-cache before splice-back and summary generation.

    Args:
        cache: The KV-cache to modify.
        start: Start position (inclusive) of the range to remove.
        end: End position (exclusive) of the range to remove.

    Returns:
        A new DynamicCache with the range removed.
    """
    if start >= end:
        return clone_cache(cache)

    out = DynamicCache()
    n = get_cache_num_layers(cache)
    for i in range(n):
        k, v = get_cache_kv(cache, i)
        new_k = torch.cat([k[:, :, :start, :], k[:, :, end:, :]], dim=2)
        new_v = torch.cat([v[:, :, :start, :], v[:, :, end:, :]], dim=2)
        out.update(new_k, new_v, i)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Prefill
# ═══════════════════════════════════════════════════════════════════════════


def prefill(model, text: str, tokenizer, device: str) -> DynamicCache:
    """Run a full forward pass on `text`, return a cloned DynamicCache."""
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    return clone_cache(outputs.past_key_values)


# ═══════════════════════════════════════════════════════════════════════════
# Generation helpers
# ═══════════════════════════════════════════════════════════════════════════


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Sample a token from logits via temperature + top-p.

    Accepts logits of shape [B, V] (already last-step) or [B, T, V] (full sequence).
    Returns shape [B, 1] of token ids.
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    cutoff[:, 1:] = cutoff[:, :-1].clone()
    cutoff[:, 0] = False
    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(-1, idx)


def generate_one_token(
    model, input_ids: torch.Tensor, past_kv: DynamicCache, position: int
):
    """Forward one new token. Returns (logits, updated_kv)."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_kv,
            position_ids=torch.tensor([[position]]).to(input_ids.device),
            use_cache=True,
        )
    return outputs.logits, outputs.past_key_values


def generate_until_stop(
    model,
    tokenizer,
    past_kv: DynamicCache,
    start_position: int,
    start_logits: torch.Tensor,
    stop_strings: list[str],
    max_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
):
    """Generate tokens until one of `stop_strings` appears in decoded text,
    or `max_tokens` is reached.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Associated tokenizer.
        past_kv: Starting KV-cache (including all previous context).
        start_position: Current token position in the cache.
        start_logits: Logits for the first token, shape [B, V] or [B, 1, V].
        stop_strings: Strings that trigger stop (checked in accumulated text).
        max_tokens: Safety cap on tokens to generate.

    Returns:
        (generated_text, updated_kv_cache, new_position)
    """
    current_kv = past_kv
    pos = start_position
    text = ""

    # First token from provided logits
    next_token = sample_token(start_logits, temperature, top_p)
    tid = next_token.item()
    if tid == tokenizer.eos_token_id:
        return text, current_kv, pos
    decoded = tokenizer.decode([tid])
    text += decoded

    for _ in range(max_tokens):
        logits, current_kv = generate_one_token(model, next_token, current_kv, pos)
        pos += 1
        next_token = sample_token(logits, temperature, top_p)
        tid = next_token.item()
        if tid == tokenizer.eos_token_id:
            break
        decoded = tokenizer.decode([tid])
        text += decoded

        # Check stop strings in accumulated text
        if any(s in text for s in stop_strings):
            break

    return text, current_kv, pos


def forward_tokens(
    model,
    tokenizer,
    text: str,
    past_kv: DynamicCache,
    position: int,
    device: str,
) -> tuple[torch.Tensor, DynamicCache, int]:
    """Forward a string through the model, appending tokens to the KV-cache.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Associated tokenizer.
        text: Text to encode and forward.
        past_kv: Current KV-cache.
        position: Current token position.
        device: Torch device string.

    Returns:
        (next_token_logits, updated_kv_cache, new_position)
    """
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_kv,
            use_cache=True,
        )
    new_kv = clone_cache(outputs.past_key_values)
    new_pos = position + input_ids.shape[1]
    return outputs.logits[:, -1, :], new_kv, new_pos


# ═══════════════════════════════════════════════════════════════════════════
# Variant Pre-computation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class VariantSet:
    """Container for pre-computed KV variants.

    Attributes:
        variants: Mapping from frozenset(revealed_fields) → DynamicCache.
        prefix_len: Number of tokens in the prefix (splice boundary).
        prompt_len: Total token count of any variant (all identical after alignment).
        masks: {field_name: token_mask_string} for the all_masked variant.
        real_pads: Per-variant padding applied to real values for length alignment.
            Keyed by frozenset(revealed_fields) → {field_name: pad_string}.
    """
    variants: dict[frozenset, DynamicCache]
    prefix_len: int
    prompt_len: int
    masks: dict[str, str]
    real_pads: dict[frozenset, dict[str, str]]
    prompts: dict[frozenset, str] = field(default_factory=dict)


class KVVariantBuilder:
    """Builds 2^N KV-cache variants, one per combination of revealed fields.

    Generalized from _build_all_prompts + prefill in method2_kvcache.py.
    Works with any set of field names via itertools.combinations.

    Alignment strategy:
      1. Create initial token-aligned masks via SensitiveRegistry.
      2. Grow each field's mask so all_masked dominates real variants in token length.
      3. For real-using variants, append trailing pads to real values to match
         all_masked's token count.
      4. All 2^N variants end up with identical total token counts.
    """

    def __init__(
        self,
        model,
        tokenizer,
        registry: SensitiveRegistry,
        prompt_builder: PromptBuilder,
        device: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.prompt_builder = prompt_builder
        self.device = device

    # ── Public API ──────────────────────────────────────────────────────

    def build_all_variants(self, user_task: str) -> VariantSet:
        """Pre-compute all 2^N variants.

        Steps:
          1. Generate aligned prompts for all 2^N field combinations.
          2. Prefill each prompt to get its KV-cache.
          3. Verify all KV shapes match.
          4. Return VariantSet.
        """
        field_names = self.registry.field_names
        n = len(field_names)

        # All field combinations: {} , {f0}, {f1}, ..., {f0,f1}, ..., {f0,...,fn}
        all_combos: list[frozenset] = []
        for r in range(n + 1):
            for combo in combinations(field_names, r):
                all_combos.append(frozenset(combo))

        # 1. Align prompts
        prompts, prefix_len, masks, real_pads = self._align_prompts(
            all_combos, user_task
        )

        # 2. Prefill each variant
        variants: dict[frozenset, DynamicCache] = {}
        for combo in all_combos:
            prompt_text = prompts[combo]
            kv = prefill(self.model, prompt_text, self.tokenizer, self.device)
            variants[combo] = kv

        # 3. Verify shape consistency
        first_k, _ = get_cache_kv(variants[frozenset()], 0)
        first_shape = first_k.shape
        for combo, kv in variants.items():
            k, _ = get_cache_kv(kv, 0)
            assert k.shape == first_shape, (
                f"Shape mismatch: all_masked {first_shape} vs combo {combo} {k.shape}"
            )

        # 4. prompt_len = sequence length (same for all variants)
        prompt_len = get_cache_seq_len(variants[frozenset()])

        return VariantSet(
            variants=variants,
            prefix_len=prefix_len,
            prompt_len=prompt_len,
            masks=masks,
            real_pads=real_pads,
            prompts={k: v for k, v in prompts.items()},
        )

    def build_single_variant(
        self,
        revealed: frozenset,
        masks: dict[str, str],
        real_pads: dict[frozenset, dict[str, str]],
        user_task: str,
    ) -> DynamicCache:
        """Dynamically prefill a variant for a combination not in the pre-computed set.

        Port of _get_or_build_kv from method2_kvcache.py.
        """
        values = {}
        pads = real_pads.get(revealed, {})
        for name in self.registry.field_names:
            if name in revealed:
                values[name] = self.registry.real(name) + pads.get(name, "")
            else:
                values[name] = masks.get(name, self.registry.token_mask(name))

        prompt = self.prompt_builder.build_full_prompt_method2(values, user_task)
        return prefill(self.model, prompt, self.tokenizer, self.device)

    # ── Internals: alignment logic ──────────────────────────────────────

    def _align_prompts(
        self,
        all_combos: list[frozenset],
        user_task: str,
    ) -> tuple[dict, int, dict, dict]:
        """Core alignment logic. Generalized from _build_all_prompts.

        Returns:
            (prompts_dict, prefix_len, masks_dict, real_pads_dict)

        prompts_dict maps frozenset(revealed_fields) → prompt_string.
        All prompts have identical total token length.
        """
        field_names = self.registry.field_names

        # Initial masks from registry
        masks = {name: self.registry.token_mask(name) for name in field_names}

        # Prefix length function (in tokens, for the prefix section only)
        def _prefix_tok_len(revealed: frozenset) -> int:
            vals = self._values_for(revealed, masks)
            prefix_text = self.prompt_builder.build_prefix(vals)
            return len(self.tokenizer.encode(prefix_text, add_special_tokens=False))

        # Full prompt length function
        def _full_len(revealed: frozenset) -> int:
            vals = self._values_for(revealed, masks)
            prompt = self.prompt_builder.build_full_prompt_method2(vals, user_task)
            return len(self.tokenizer.encode(prompt, add_special_tokens=False))

        # Step 1: Grow each field's mask so all_masked dominates single-real variants
        for f in field_names:
            target = _full_len(frozenset({f}))
            self._grow_mask(masks, f, target, _full_len)

        # Step 2: Grow masks to cover the all-revealed variant
        all_revealed = frozenset(field_names)
        target = _full_len(all_revealed)
        for f in field_names:
            cur = _full_len(frozenset())
            if cur >= target:
                break
            self._grow_mask(masks, f, target, _full_len)
            target = _full_len(all_revealed)

        base_len = _full_len(frozenset())

        # Step 3: Fit real-value pads for each variant to match base_len
        real_pads: dict[frozenset, dict[str, str]] = {}
        for combo in all_combos:
            if not combo:
                real_pads[combo] = {}
                continue
            ok, pads = self._fit_real_pads(combo, base_len, masks, _full_len)
            real_pads[combo] = pads
            if not ok:
                pass  # alignment failure is non-fatal; variant may have slight offset

        # Step 4: Build final prompts with captured pads
        prompts: dict[frozenset, str] = {}
        for combo in all_combos:
            vals = self._values_for(combo, masks)
            pads = real_pads.get(combo, {})
            for f in combo:
                vals[f] = vals[f] + pads.get(f, "")
            prompts[combo] = self.prompt_builder.build_full_prompt_method2(
                vals, user_task
            )

        return prompts, _prefix_tok_len(frozenset()), masks, real_pads

    def _values_for(self, revealed: frozenset, masks: dict[str, str]) -> dict[str, str]:
        """Build {field_name: value} dict for a given revealed set."""
        result = {}
        for name in self.registry.field_names:
            if name in revealed:
                result[name] = self.registry.real(name)
            else:
                result[name] = masks[name]
        return result

    def _grow_mask(
        self,
        masks: dict[str, str],
        field: str,
        want_at_least: int,
        full_len_fn: Callable[[frozenset], int],
        max_steps: int = 256,
    ) -> None:
        """Greedily grow masks[field] until the all_masked variant's token count
        reaches `want_at_least`.

        Port of the `grow_mask` inner function from _build_all_prompts.
        """
        palette = list(dict.fromkeys(
            list(self.registry.palettes.space_prefixed or [])
            + list(self.registry.palettes.regular)
        )) or ["_"]

        def _cur_base():
            return full_len_fn(frozenset())

        for _ in range(max_steps):
            base = _cur_base()
            if base >= want_at_least:
                return
            grew = False
            for ch in palette:
                for sep in (" ", ""):
                    trial = masks[field] + sep + ch
                    old = masks[field]
                    masks[field] = trial
                    new_base = _cur_base()
                    if new_base > base:
                        grew = True
                        break
                    masks[field] = old
                if grew:
                    break
            if not grew:
                return

    def _fit_real_pads(
        self,
        combo: frozenset,
        base_len: int,
        masks: dict[str, str],
        full_len_fn: Callable[[frozenset], int],
    ) -> tuple[bool, dict[str, str]]:
        """Greedily fit padding to real values so this variant's total token
        count matches base_len.

        Port of the `fit_real_pads` inner function from _build_all_prompts.
        """
        palette = list(dict.fromkeys(
            list(self.registry.palettes.space_prefixed or [])
            + list(self.registry.palettes.regular)
        )) or ["_"]

        pads = {f: "" for f in combo}
        # Save original mask values (fields in combo use real, others use mask)
        orig_masks = dict(masks)

        def _len() -> int:
            return full_len_fn(combo)

        cur = _len()
        if cur > base_len:
            return False, pads
        if cur == base_len:
            return True, pads

        for _ in range(512):
            cur = _len()
            if cur == base_len:
                return True, pads
            if cur > base_len:
                return False, pads
            grew = False
            for f in list(combo):
                old = pads[f]
                # Try +1 step first
                best = None
                for ch in palette:
                    for sep in (" ", ""):
                        trial = old + sep + ch
                        pads[f] = trial
                        nl = _len()
                        if nl == base_len:
                            return True, pads
                        if nl == cur + 1:
                            best = trial
                            break
                        pads[f] = old
                    if best is not None:
                        break
                if best is not None:
                    pads[f] = best
                    grew = True
                    break
                # Otherwise any positive non-overshoot
                for ch in palette:
                    for sep in (" ", ""):
                        trial = old + sep + ch
                        pads[f] = trial
                        nl = _len()
                        if nl == base_len:
                            return True, pads
                        if cur < nl < base_len:
                            grew = True
                            break
                        pads[f] = old
                    if grew:
                        break
                if grew:
                    break
            if not grew:
                return False, pads
        return _len() == base_len, pads
