"""
Method 2: Local Model KV-Cache Approach — Prefix KV Splice

Core idea:
  - Sensitive data is placed at the VERY BEGINNING of the input (the "prefix").
  - At init time, we prefill 8 prompt variants to get 8 complete KV-caches:
      kv_all    — all fields masked
      kv_api    — only api_key unmasked
      kv_phone  — only phone unmasked
      kv_email  — only email unmasked
      kv_api_phone   — api_key + phone unmasked
      kv_api_email   — api_key + email unmasked
      kv_phone_email — phone + email unmasked
      kv_all_three   — all three fields unmasked
  - During normal operation we use kv_all.
  - When the AI requests a field, we SPLICE the prefix portion of the
    corresponding unmasked KV into the current KV. The suffix (task
    description, generated tokens) stays UNCHANGED.
  - When the AI signals completion, we splice the all-masked prefix back,
    erasing sensitive info from the model's state.
"""

from __future__ import annotations

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from config import (
    SENSITIVE_REQUEST_PATTERN,
    SENSITIVE_CLEAR_TOKEN,
    SENSITIVE_FIELDS_CHAR,
    LOCAL_MODEL_NAME,
    MAX_GENERATE_TOKENS,
)
from utils import (
    print_header,
    print_info,
    print_sensitive,
    print_system,
    build_full_prompt,
    DEMO_TASK,
)


# ── Token-Level Masker ──────────────────────────────────────────────────────

class TokenLevelMasker:
    """Creates masks that produce the SAME number of tokens as the real value.

    This is critical: when masked and real produce identical token counts,
    all 4 prompt variants have the same total length, so KV-cache tensors
    have identical shapes and can be spliced freely.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_palette = self._find_single_token_chars()
        self.space_palette = self._find_single_token_space_chars()

    def _find_single_token_chars(self) -> list[str]:
        candidates = [
            "█", "▓", "▒", "░", "▇", "▆", "▅", "▄", "▃", "▂", "▁",
            "■", "□", "▪", "▫", "▮", "▯", "▰", "▱", "▲", "△",
            "▴", "▵", "▶", "▷", "▸", "▹", "►", "▻", "▼", "▽",
            "▾", "▿", "◀", "◁", "◂", "◃", "◄", "◅", "◆", "◇",
            "◈", "◉", "◊", "○", "◌", "◍", "◎", "●",
            "◐", "◑", "◒", "◓", "◔", "◕", "◖", "◗",
            "◘", "◙", "◚", "◛", "◜", "◝", "◞", "◟",
            "◠", "◡", "◢", "◣", "◤", "◥", "◦", "◧",
            "◨", "◩", "◪", "◫", "◬", "◭", "◮", "◯",
        ]
        result = []
        for ch in candidates:
            ids = self.tokenizer.encode(ch, add_special_tokens=False)
            if len(ids) == 1:
                result.append(ch)
                if len(result) >= 20:
                    break
        if not result:
            result = ["_"]
        return result

    def _find_single_token_space_chars(self) -> list[str]:
        """Find chars that remain a single token when prefixed by a space."""
        result = []
        for ch in self.mask_palette:
            ids = self.tokenizer.encode(" " + ch, add_special_tokens=False)
            if len(ids) == 1:
                result.append(ch)
        return result

    def append_safe_tokens(self, mask_str: str, count: int) -> str:
        """Append tokens that are likely to add exactly one token each in context."""
        palette = self.space_palette or self.mask_palette
        for i in range(count):
            ch = palette[i % len(palette)]
            if palette is self.space_palette:
                mask_str += " " + ch
            else:
                mask_str += ch
        return mask_str

    def mask_value(self, real_value: str) -> tuple[str, list[int], list[int]]:
        """Create a mask string that tokenizes to the same length as real_value."""
        real_ids = self.tokenizer.encode(real_value, add_special_tokens=False)
        target_len = len(real_ids)

        # Alternating palette chars → each is 1 token, no BPE merging
        parts = []
        for i in range(target_len):
            parts.append(self.mask_palette[i % len(self.mask_palette)])
        mask_str = "".join(parts)
        mask_ids = self.tokenizer.encode(mask_str, add_special_tokens=False)

        attempt = 0
        while len(mask_ids) != target_len and attempt < 20:
            attempt += 1
            if len(mask_ids) < target_len:
                mask_str += self.mask_palette[attempt % len(self.mask_palette)]
            else:
                mask_str = mask_str[:-1]
            mask_ids = self.tokenizer.encode(mask_str, add_special_tokens=False)

        if len(mask_ids) != target_len:
            mask_str = "*" * len(real_value)
            mask_ids = self.tokenizer.encode(mask_str, add_special_tokens=False)
            print_info(
                f"警告: 无法为 '{real_value}' 创建精确 token 对齐的 mask。"
                f"目标={target_len}, 实际={len(mask_ids)}。"
            )

        return mask_str, real_ids, mask_ids


# ── Prompt Construction ─────────────────────────────────────────────────────

PREFIX_TEMPLATE = (
    "敏感信息:\n"
    "- api_key: {api_key}\n"
    "- phone: {phone}\n"
    "- email: {email}\n\n"
)

SYSTEM_PROTOCOL = (
    "你是一个AI助手。你可以在对话开头看到一些敏感信息（当前为隐藏状态）。\n"
    "当你需要使用某项敏感信息时，请输出精确格式：\n"
    "  <<SENSITIVE_REQUEST:api_key>> 或 <<SENSITIVE_REQUEST:phone>> 或 <<SENSITIVE_REQUEST:email>>\n"
    "不要输出“字段名”这样的占位词，必须使用上述三者之一。\n"
    "重要：必须先请求所有字段，再输出发送提示：\n"
    "  Step 1 先依次请求 email、phone、api_key（允许在同一轮连续输出三个请求）输出请求之后直接结束本轮输出，等待下一轮输入！！！；\n"
    "  Step 2 得到输入，所需敏感信息都拿到后，按相应格式输出发送提示，输出所有发送提示之后直接结束本轮输出，等待下一轮输入！！！；\n"
    "  Step 3 最后输出 <<SENSITIVE_CLEAR>>。\n"
    "禁止在请求完成之前输出邮件/短信格式或 <<SENSITIVE_CLEAR>>。\n"
    "当你的任务完成、不再需要敏感信息时，请输出 <<SENSITIVE_CLEAR>> 来触发敏感信息清除。\n"
    "注意：不要编造敏感信息的具体值，必须通过 <<SENSITIVE_REQUEST:...>> 来获取。\n"
    "格式说明：每个敏感字段的值末尾可能含有用于对齐的空格或特殊字符（如 ▇/■/◆ 等），"
    "这些是 padding，不属于真实数据。使用真实值时请只取前面非 padding 部分。\n"
    "输出规范：当获得敏感信息后，必须按邮件/短信格式输出发送提示：\n"
    "- 邮件格式示例：\n"
    "  [发送邮件]\n"
    "  To: <完整邮箱地址>\n"
    "  Subject: 会议通知\n"
    "  Body: 会议改到明天下午3点。\n"
    "- 短信格式示例：\n"
    "  [发送短信]\n"
    "  To: <完整手机号>\n"
    "  Body: 请查收邮件。\n"
    "以上格式中的 To 必须使用解密后的完整敏感信息。"
)


def _build_one_prompt(field_values: dict[str, str], user_task: str) -> str:
    """Build a complete prompt string given field values (real or masked)."""
    prefix = PREFIX_TEMPLATE.format(**field_values)
    suffix = build_full_prompt(SYSTEM_PROTOCOL, user_task)
    return prefix + suffix


def _build_all_prompts(masker, sensitive_fields, user_task):
    """Build all 8 prompt variants with identical prefix token length.

    Strategy (per user's design):
      1. Initial mask: per-field mask whose token count matches the real
         value (via mask_value).
      2. Grow each revealable field's mask so that all_masked is at least
         as long (in tokens) as any variant where that field is real.
         Mask growth ONLY affects variants where the field is masked.
      3. For each real-using variant, append a trailing pad (space + palette
         char) AFTER the real value to bring its prefix length up to the
         all_masked target. Real-pad ONLY affects that real variant.

    Both mask growth and real-pad are slot-local, so the 8 variants
    can be aligned independently.
    """
    tokenizer = masker.tokenizer

    masks = {}
    for name, info in sensitive_fields.items():
        mask_str, _, _ = masker.mask_value(info["real"])
        masks[name] = mask_str

    revealable = ["api_key", "phone", "email"]
    real_pads: dict[str, str] = {f: "" for f in revealable}

    def values_for(real_set):
        out = {}
        for n in sensitive_fields:
            if n in real_set:
                out[n] = sensitive_fields[n]["real"] + real_pads.get(n, "")
            else:
                out[n] = masks[n]
        return out

    def plen(real_set):
        return len(tokenizer.encode(
            PREFIX_TEMPLATE.format(**values_for(real_set)),
            add_special_tokens=False,
        ))

    palette = list(dict.fromkeys(
        list(masker.space_palette or []) + list(masker.mask_palette)
    )) or ["_"]

    # ── Step 1: Grow each mask so all_masked dominates the corresponding
    #            single-real variant. Then ensure it dominates `both`.
    def grow_mask(field: str, want_at_least: int, max_steps: int = 256) -> None:
        """Append chars to masks[field] until plen({}) >= want_at_least."""
        for _ in range(max_steps):
            base = plen(frozenset())
            if base >= want_at_least:
                return
            grew = False
            for ch in palette:
                for sep in (" ", ""):
                    trial = masks[field] + sep + ch
                    old = masks[field]
                    masks[field] = trial
                    new_base = plen(frozenset())
                    if new_base > base:
                        grew = True
                        break
                    masks[field] = old
                if grew:
                    break
            if not grew:
                return

    for f in revealable:
        target = plen(frozenset({f}))
        grow_mask(f, target)

    target_both = plen(frozenset(revealable))
    for f in revealable:
        if plen(frozenset()) >= target_both:
            break
        grow_mask(f, target_both)
        target_both = plen(frozenset(revealable))  # may shift slightly

    base_len = plen(frozenset())

    # ── Step 2: For each real variant, fit real_pads so prefix == base_len.
    # Real-pad goes only on real values; masking variants are unaffected.
    def fit_real_pads(real_set: frozenset) -> bool:
        # Reset only the fields in this variant
        for f in real_set:
            real_pads[f] = ""
        cur = plen(real_set)
        if cur > base_len:
            return False  # base_len wasn't truly the max — caller will handle
        if cur == base_len:
            return True
        # Greedy: extend real_pads[f] for some f in real_set
        for _ in range(512):
            cur = plen(real_set)
            if cur == base_len:
                return True
            if cur > base_len:
                return False
            grew = False
            for f in list(real_set):
                old = real_pads[f]
                # Try +1 step first
                best = None
                for ch in palette:
                    for sep in (" ", ""):
                        trial = old + sep + ch
                        real_pads[f] = trial
                        nl = plen(real_set)
                        if nl == base_len:
                            return True
                        if nl == cur + 1:
                            best = (trial, nl)
                            break
                        real_pads[f] = old
                    if best is not None:
                        break
                if best is not None:
                    real_pads[f] = best[0]
                    grew = True
                    break
                # Otherwise any positive non-overshoot
                for ch in palette:
                    for sep in (" ", ""):
                        trial = old + sep + ch
                        real_pads[f] = trial
                        nl = plen(real_set)
                        if nl == base_len:
                            return True
                        if cur < nl < base_len:
                            grew = True
                            break
                        real_pads[f] = old
                    if grew:
                        break
                if grew:
                    break
            if not grew:
                return False
        return plen(real_set) == base_len

    per_variant_real_pads: dict[str, dict[str, str]] = {}
    for vname, rs in [
        ("all_masked",  frozenset()),
        ("api",         frozenset({"api_key"})),
        ("phone",       frozenset({"phone"})),
        ("email",       frozenset({"email"})),
        ("api_phone",   frozenset({"api_key", "phone"})),
        ("api_email",   frozenset({"api_key", "email"})),
        ("phone_email", frozenset({"phone", "email"})),
        ("all_three",   frozenset(revealable)),
    ]:
        if not rs:
            per_variant_real_pads[vname] = {}
            continue
        ok = fit_real_pads(rs)
        if not ok:
            print_info(
                f"警告: 变体 '{vname}' 无法对齐到 base_len={base_len}, "
                f"当前长度={plen(rs)}"
            )
        per_variant_real_pads[vname] = {f: real_pads[f] for f in rs}

    # Build the actual prompts using the captured per-variant pads.
    def build_prompt(vname: str, rs: frozenset) -> str:
        for f in revealable:
            real_pads[f] = per_variant_real_pads[vname].get(f, "")
        return _build_one_prompt(values_for(rs), user_task)

    prompts = {
        "all_masked":  build_prompt("all_masked",  frozenset()),
        "api":         build_prompt("api",         frozenset({"api_key"})),
        "phone":       build_prompt("phone",       frozenset({"phone"})),
        "email":       build_prompt("email",       frozenset({"email"})),
        "api_phone":   build_prompt("api_phone",   frozenset({"api_key", "phone"})),
        "api_email":   build_prompt("api_email",   frozenset({"api_key", "email"})),
        "phone_email": build_prompt("phone_email", frozenset({"phone", "email"})),
        "all_three":   build_prompt("all_three",   frozenset(revealable)),
    }
    prefix_len = base_len

    # Stash per-variant pad info so the runner can reproduce real-padded
    # values during dynamic prefill (Phase 2 fallbacks).
    return prompts, prefix_len, masks, per_variant_real_pads


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks before parsing control tokens."""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


# ── KV-Cache Operations (DynamicCache-native) ───────────────────────────────
#
# We use DynamicCache throughout. Different transformers versions expose
# the per-layer key/value tensors slightly differently, so the helpers below
# normalize access:
#   - new API (4.45+): cache.layers[i].keys / .values
#   - legacy API:      cache.key_cache[i] / cache.value_cache[i]


def _cache_num_layers(cache) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "layers") and cache.layers is not None:
        return len(cache.layers)
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    return 0


def _cache_get_kv(cache, i):
    if hasattr(cache, "layers") and cache.layers is not None:
        layer = cache.layers[i]
        return layer.keys, layer.values
    return cache.key_cache[i], cache.value_cache[i]


def _cache_set_kv(cache, i, k, v) -> None:
    if hasattr(cache, "layers") and cache.layers is not None:
        cache.layers[i].keys = k
        cache.layers[i].values = v
    else:
        cache.key_cache[i] = k
        cache.value_cache[i] = v


def clone_cache(cache):
    """Deep-copy a DynamicCache (tensors detached + cloned)."""
    if cache is None:
        return None
    new = DynamicCache()
    n = _cache_num_layers(cache)
    for i in range(n):
        k, v = _cache_get_kv(cache, i)
        new.update(k.detach().clone(), v.detach().clone(), i)
    return new


def splice_prefix(source_cache, target_cache, prefix_len: int):
    """Replace the first `prefix_len` positions of target with source's prefix.

    Only the prefix portion is swapped; positions >= prefix_len stay unchanged.
    Operates on DynamicCache objects directly — no legacy tuple round-trip.
    """
    out = clone_cache(target_cache)
    n = _cache_num_layers(out)
    for i in range(n):
        src_k, src_v = _cache_get_kv(source_cache, i)
        tgt_k, tgt_v = _cache_get_kv(out, i)
        new_k = torch.cat(
            [src_k[:, :, :prefix_len, :], tgt_k[:, :, prefix_len:, :]], dim=2
        )
        new_v = torch.cat(
            [src_v[:, :, :prefix_len, :], tgt_v[:, :, prefix_len:, :]], dim=2
        )
        _cache_set_kv(out, i, new_k, new_v)
    return out


def prefill(model, text: str, tokenizer, device):
    """Run a full forward pass on `text`, return a cloned DynamicCache."""
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    return clone_cache(outputs.past_key_values)


# ── Generation Helpers ──────────────────────────────────────────────────────


def sample_token(logits, temperature: float = 0.6, top_p: float = 0.9):
    """Sample a token from logits via temperature + top-p.

    Accepts logits of shape [B, V] (already last-step) or [B, T, V] (full).
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


def generate_one_token(model, input_ids, past_kv, position):
    """Forward one new token, always passing a DynamicCache."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_kv,
            position_ids=torch.tensor([[position]]).to(input_ids.device),
            use_cache=True,
        )
    return outputs.logits, outputs.past_key_values


# ── Main Method 2 Runner ────────────────────────────────────────────────────

class KVCacheSensitiveRunner:

    def __init__(self, model_name=LOCAL_MODEL_NAME, device="auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sensitive_fields = SENSITIVE_FIELDS_CHAR
        self._load_model()

    def _load_model(self):
        print_info(f"加载模型: {self.model_name} (device={self.device}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        print_info("模型加载完成。")

        self.masker = TokenLevelMasker(self.tokenizer)

    # ── Main Demo ────────────────────────────────────────────────────────

    def run_demo(self):
        print_header("方法二：本地模型 KV-Cache — Prefix KV Splice")

        # ═══════════════════════════════════════════════════════════════
        # Phase 0: Pre-compute 8 KV-cache variants
        # ═══════════════════════════════════════════════════════════════
        print_header("阶段 0：预计算 8 个 KV-Cache 变体")

        prompts, prefix_len, masks, real_pads_by_variant = _build_all_prompts(
            self.masker, self.sensitive_fields, DEMO_TASK
        )
        self._masks = masks
        self._real_pads_by_variant = real_pads_by_variant

        # Verify all prompts have the same token length
        lengths = {}
        for name, text in prompts.items():
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            lengths[name] = len(ids)
        prompt_len = lengths["all_masked"]
        print_info(f"提示词总长度: {prompt_len} tokens, 前缀长度: {prefix_len} tokens")
        print_info(f"各变体长度: {lengths}")
        assert len(set(lengths.values())) == 1, \
            f"Token counts differ! {lengths}"

        # Prefill all 8 variants
        print_info("Prefill 全保密 (all_masked)...")
        kv_all = prefill(self.model, prompts["all_masked"], self.tokenizer, self.device)

        print_info("Prefill A公开 (api_key)...")
        kv_api = prefill(self.model, prompts["api"], self.tokenizer, self.device)

        print_info("Prefill B公开 (phone)...")
        kv_phone = prefill(self.model, prompts["phone"], self.tokenizer, self.device)

        print_info("Prefill C公开 (email)...")
        kv_email = prefill(self.model, prompts["email"], self.tokenizer, self.device)

        print_info("Prefill AB公开 (api_key + phone)...")
        kv_api_phone = prefill(self.model, prompts["api_phone"], self.tokenizer, self.device)

        print_info("Prefill AC公开 (api_key + email)...")
        kv_api_email = prefill(self.model, prompts["api_email"], self.tokenizer, self.device)

        print_info("Prefill BC公开 (phone + email)...")
        kv_phone_email = prefill(self.model, prompts["phone_email"], self.tokenizer, self.device)

        print_info("Prefill ABC公开 (all three)...")
        kv_all_three = prefill(self.model, prompts["all_three"], self.tokenizer, self.device)

        # Mapping: frozenset(revealed_fields) → pre-computed KV
        kv_variants = {
            frozenset():                               kv_all,
            frozenset({"api_key"}):                    kv_api,
            frozenset({"phone"}):                      kv_phone,
            frozenset({"email"}):                      kv_email,
            frozenset({"api_key", "phone"}):           kv_api_phone,
            frozenset({"api_key", "email"}):           kv_api_email,
            frozenset({"phone", "email"}):             kv_phone_email,
            frozenset({"api_key", "phone", "email"}):  kv_all_three,
        }

        # Verify shapes match
        first_k, _ = _cache_get_kv(kv_all, 0)
        first_shape = first_k.shape
        for name, kv in [
            ("api", kv_api), ("phone", kv_phone), ("email", kv_email),
            ("api_phone", kv_api_phone), ("api_email", kv_api_email),
            ("phone_email", kv_phone_email), ("all_three", kv_all_three),
        ]:
            k, _ = _cache_get_kv(kv, 0)
            assert k.shape == first_shape, \
                f"Shape mismatch: all_masked {first_shape} vs {name} {k.shape}"

        print_system("8 个 KV-Cache 变体预计算完成，shape 一致。\n")
        print_info(
            f"KV shape 示例: [batch={first_shape[0]}, heads={first_shape[1]}, "
            f"seq={first_shape[2]}, dim={first_shape[3]}]"
        )
        print_system(
            "关键：prefix 部分占据 tokens [0..{})，suffix 占据 tokens [{}..{})\n".format(
                prefix_len, prefix_len, prompt_len
            )
        )

        # Logits for first token from all_masked
        all_ids = self.tokenizer.encode(prompts["all_masked"], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=all_ids, use_cache=True)
        next_logits = outputs.logits[:, -1, :]

        revealed_fields: set = set()

        # ═══════════════════════════════════════════════════════════════
        # Phase 1: Generate with all-masked KV
        # ═══════════════════════════════════════════════════════════════
        print_header("阶段 1：使用全保密 KV 推理")

        gen_text, current_kv, current_pos = self._generate_until_special(
            next_logits=next_logits,
            past_kv=clone_cache(kv_all),
            start_position=prompt_len,
            stop_on_request=False,
            stop_on_clear=False,
            stop_on_im_end=True,
            max_tokens=MAX_GENERATE_TOKENS,
        )

        parsed_text = _strip_think_blocks(gen_text or "")
        if SENSITIVE_CLEAR_TOKEN in parsed_text:
            print_system("模型已输出清除指令，demo 结束。")
            return

        req_match = SENSITIVE_REQUEST_PATTERN.search(parsed_text)
        if not req_match:
            print_system("模型未请求敏感信息，demo 结束。")
            return

        # Extract ALL requested fields (model may request multiple at once)
        req_fields = self._extract_requested_fields(parsed_text)
        print_sensitive(f"\n>>> AI 请求敏感信息: {', '.join(req_fields)}")

        # ═══════════════════════════════════════════════════════════════
        # Phase 2: Splice unmasked prefix into current KV
        # ═══════════════════════════════════════════════════════════════
        self._phase2_unmask_and_generate(
            req_fields, revealed_fields, kv_variants,
            current_kv, current_pos, prefix_len, prompt_len
        )
        current_kv = self._current_kv
        current_pos = self._current_pos

        # ═══════════════════════════════════════════════════════════════
        # Phase 3: Splice all-masked prefix back
        # ═══════════════════════════════════════════════════════════════
        print_header("阶段 3：恢复全保密 KV（Splice 回 all_masked prefix）")

        restored_kv = splice_prefix(kv_all, current_kv, prefix_len)

        print_system("已将 prefix 替换回全保密状态。")
        print_system("suffix 部分（任务描述 + 已生成的 token）KV 未变。\n")

        summary_text = (
            f"\n<|im_start|>user\n"
            f"任务已完成。请总结你做了什么（不要再使用敏感信息的具体值）。\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        summary_ids = self.tokenizer.encode(summary_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=summary_ids,
                past_key_values=restored_kv,
                use_cache=True,
            )
        final_kv = clone_cache(outputs.past_key_values)
        final_pos = current_pos + summary_ids.shape[1]

        print_system("验证：模型不再能访问敏感信息。\n")
        _, _, _ = self._generate_until_special(
            next_logits=outputs.logits[:, -1, :],
            past_kv=final_kv,
            start_position=final_pos,
            stop_on_request=True,
            stop_on_clear=False,
            stop_on_im_end=True,
            max_tokens=MAX_GENERATE_TOKENS,
        )

        print()
        print_system("Demo 完成。")
        print_system(
            "关键机制："
            "\n  1. 预计算 8 个完整 KV: 全保密 / A / B / C / AB / AC / BC / ABC"
            "\n  2. 正常推理使用全保密 KV"
            "\n  3. 请求敏感信息时 splice_prefix(kv_variants[revealed], current_kv, prefix_len)"
            "\n     → 仅替换 prefix 部分，suffix 完全不变"
            "\n  4. 任务完成后 splice_prefix(kv_all, current_kv, prefix_len)"
            "\n     → prefix 恢复全保密，敏感信息从模型状态消失"
        )

    # ── Phase 2 Helper ───────────────────────────────────────────────────

    # ── Variant description lookup ────────────────────────────────────────

    _VARIANT_DESC = {
        frozenset(): "全保密",
        frozenset({"api_key"}): "A公开 (api_key)",
        frozenset({"phone"}): "B公开 (phone)",
        frozenset({"email"}): "C公开 (email)",
        frozenset({"api_key", "phone"}): "AB公开 (api_key + phone)",
        frozenset({"api_key", "email"}): "AC公开 (api_key + email)",
        frozenset({"phone", "email"}): "BC公开 (phone + email)",
        frozenset({"api_key", "phone", "email"}): "ABC公开 (all three)",
    }

    @staticmethod
    def _extract_requested_fields(text: str) -> list[str]:
        """Extract all unique requested fields from AI text, preserving order."""
        seen = set()
        fields = []
        for match in SENSITIVE_REQUEST_PATTERN.finditer(text):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                fields.append(name)
        return fields

    # ── Phase 2 Helper ───────────────────────────────────────────────────

    def _phase2_unmask_and_generate(
        self, field_names, revealed_fields, kv_variants,
        current_kv, current_pos, prefix_len, prompt_len
    ):
        """Splice the appropriate unmasked prefix and continue generation.

        Accepts a list of field_names (model may request multiple at once).
        All requested fields are revealed together in a single splice,
        using the pre-computed KV variant matching the full revealed set.
        """
        # Reveal all requested fields at once
        for f in field_names:
            revealed_fields.add(f)
        rs = frozenset(revealed_fields)

        if rs in kv_variants:
            source_kv = kv_variants[rs]
            state_desc = self._VARIANT_DESC.get(rs, f"已解密: {revealed_fields}")
        else:
            source_kv = self._get_or_build_kv(revealed_fields, prompt_len)
            state_desc = f"已解密(动态): {revealed_fields}"

        print_header(f"阶段 2：Splice Prefix → {state_desc}")

        spliced_kv = splice_prefix(source_kv, current_kv, prefix_len)

        print_system(
            f"splice_prefix(kv, current_kv, prefix_len={prefix_len})"
        )
        print_system(
            f"→ tokens [0..{prefix_len}) 被替换为 {state_desc} 的 prefix KV"
        )
        print_system(
            f"→ tokens [{prefix_len}..{current_pos}) 保持不变\n"
        )

        # Build hint mentioning all revealed fields
        fields_str = "、".join(field_names)
        hint = (
            f"\n<|im_end|>\n"
            f"<|im_start|>user\n"
            f"[系统提示: {fields_str} 已解密，现在可以使用真实值。"
            f"请完成任务并在完成后输出 {SENSITIVE_CLEAR_TOKEN}]\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        hint_ids = self.tokenizer.encode(hint, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=hint_ids,
                past_key_values=spliced_kv,
                use_cache=True,
            )
        current_kv = clone_cache(outputs.past_key_values)
        current_pos = current_pos + hint_ids.shape[1]

        print_system(f"模型现在可以访问 {fields_str} 的真实值。\n")
        gen_text, current_kv, current_pos = self._generate_until_special(
            next_logits=outputs.logits[:, -1, :],
            past_kv=current_kv,
            start_position=current_pos,
            stop_on_request=False,
            stop_on_clear=False,
            stop_on_im_end=True,
            max_tokens=MAX_GENERATE_TOKENS * 2,
        )

        while gen_text:
            parsed_text = _strip_think_blocks(gen_text)
            if SENSITIVE_CLEAR_TOKEN in parsed_text:
                break
            req2 = SENSITIVE_REQUEST_PATTERN.search(parsed_text)
            if not req2:
                break
            field2 = req2.group(1)
            if field2 in revealed_fields:
                break
            print_sensitive(f"\n>>> AI 请求额外敏感信息: {field2}")
            revealed_fields.add(field2)
            rs2 = frozenset(revealed_fields)

            if rs2 in kv_variants:
                source_kv = kv_variants[rs2]
                state_desc = self._VARIANT_DESC.get(rs2, f"已解密: {revealed_fields}")
            else:
                source_kv = self._get_or_build_kv(revealed_fields, prompt_len)
                state_desc = f"已解密(动态): {revealed_fields}"

            spliced_kv = splice_prefix(source_kv, current_kv, prefix_len)
            print_system(f"splice_prefix → {state_desc}")

            hint2 = (
                f"\n<|im_end|>\n"
                f"<|im_start|>user\n"
                f"[{field2} 也已解密。请继续完成任务。]\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            hint_ids2 = self.tokenizer.encode(hint2, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=hint_ids2,
                    past_key_values=spliced_kv,
                    use_cache=True,
                )
            current_kv = clone_cache(outputs.past_key_values)
            current_pos = current_pos + hint_ids2.shape[1]

            gen_text, current_kv, current_pos = self._generate_until_special(
                next_logits=outputs.logits[:, -1, :],
                past_kv=current_kv,
                start_position=current_pos,
                stop_on_request=False,
                stop_on_clear=False,
                stop_on_im_end=True,
                max_tokens=MAX_GENERATE_TOKENS * 2,
            )

        self._current_kv = current_kv
        self._current_pos = current_pos

    def _get_or_build_kv(self, revealed_fields, prompt_len):
        """Build and prefill a KV for an arbitrary field combination.

        Uses the precomputed masks and real-pads so that the resulting
        prompt has the SAME prefix token length as kv_all (otherwise
        splice_prefix would mismatch and corrupt suffix positions).
        """
        rs = frozenset(revealed_fields)
        variant_key = None
        if rs == frozenset():
            variant_key = "all_masked"
        elif rs == frozenset({"api_key"}):
            variant_key = "api"
        elif rs == frozenset({"phone"}):
            variant_key = "phone"
        elif rs == frozenset({"email"}):
            variant_key = "email"
        elif rs == frozenset({"api_key", "phone"}):
            variant_key = "api_phone"
        elif rs == frozenset({"api_key", "email"}):
            variant_key = "api_email"
        elif rs == frozenset({"phone", "email"}):
            variant_key = "phone_email"
        elif rs == frozenset({"api_key", "phone", "email"}):
            variant_key = "all_three"

        pads = (self._real_pads_by_variant.get(variant_key, {})
                if variant_key else {})

        values = {}
        for name, info in self.sensitive_fields.items():
            if name in revealed_fields:
                values[name] = info["real"] + pads.get(name, "")
            else:
                values[name] = self._masks[name]
        prompt = _build_one_prompt(values, DEMO_TASK)
        return prefill(self.model, prompt, self.tokenizer, self.device)

    # ── Generation Loop ──────────────────────────────────────────────────

    def _generate_until_special(
        self, next_logits, past_kv, start_position,
        stop_on_request, stop_on_clear, stop_on_im_end, max_tokens,
    ):
        current_kv = past_kv
        pos = start_position
        text = ""

        # First token comes from the logits passed in.
        next_token = sample_token(next_logits)  # shape [1, 1]
        tid = next_token.item()
        decoded = self.tokenizer.decode([tid])
        text += decoded

        if tid == self.tokenizer.eos_token_id:
            print(text)
            return text, current_kv, pos

        for _ in range(max_tokens):
            logits, current_kv = generate_one_token(
                self.model, next_token, current_kv, pos
            )
            pos += 1
            next_token = sample_token(logits)  # shape [1, 1]
            tid = next_token.item()
            decoded = self.tokenizer.decode([tid])
            text += decoded

            if tid == self.tokenizer.eos_token_id:
                break

            if stop_on_im_end and "<|im_end|>" in text:
                break

        print(text)
        return text, current_kv, pos


# ── Standalone runner ───────────────────────────────────────────────────────

def run_method2(model_name=LOCAL_MODEL_NAME, device="auto"):
    runner = KVCacheSensitiveRunner(model_name=model_name, device=device)
    runner.run_demo()


if __name__ == "__main__":
    run_method2()
