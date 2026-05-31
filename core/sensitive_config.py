"""
Sensitive Registry — runtime definition of sensitive data fields.

Replaces the hardcoded SENSITIVE_FIELDS_CHAR in config.py with a user-configurable
registry. Supports both character-level masks (for Method 1 string replacement)
and token-level masks (for Method 2 KV-cache alignment).

Usage:
    # Method 1 (no tokenizer needed):
    registry = SensitiveRegistry({"api_key": "sk-abc123", "phone": "13800138000"})

    # Method 2 (tokenizer required for token-aligned masks):
    registry = SensitiveRegistry(
        {"api_key": "sk-abc123", "phone": "13800138000"},
        tokenizer=tokenizer,
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ── Special token patterns ──────────────────────────────────────────────────

SENSITIVE_REQUEST_PATTERN = re.compile(r"<<SENSITIVE_REQUEST:(\w+)>>")
SENSITIVE_CLEAR_TOKEN = "<<SENSITIVE_CLEAR>>"


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class SensitiveField:
    """A single sensitive field definition."""
    name: str
    real_value: str
    description: str


@dataclass
class MaskPalettes:
    """Characters that tokenize to exactly 1 token (for token-level masking)."""
    regular: list[str] = field(default_factory=list)
    space_prefixed: list[str] = field(default_factory=list)


# ── Main class ──────────────────────────────────────────────────────────────

class SensitiveRegistry:
    """Holds user-defined sensitive fields with both char-level and
    token-level masks.

    Character-level masks (always built):
        Replace non-format characters with '*', preserving '-' / '@' / '.'.
        Used by Method 1 for same-length text replacement.

    Token-level masks (built only if tokenizer provided):
        Each real token is replaced by a single-token Unicode block char.
        The mask string has the same number of tokens as the real value,
        which is critical for KV-cache shape alignment in Method 2.
    """

    def __init__(
        self,
        fields: dict[str, str],
        descriptions: Optional[dict[str, str]] = None,
        tokenizer=None,
    ):
        """Create a sensitive registry.

        Args:
            fields: {field_name: real_value} mapping.
            descriptions: Optional {field_name: human_readable_description}.
            tokenizer: Optional HuggingFace tokenizer for token-level masks.
        """
        desc = descriptions or {}

        self._fields: dict[str, SensitiveField] = {}
        for name, value in fields.items():
            self._fields[name] = SensitiveField(
                name=name,
                real_value=value,
                description=desc.get(name, f"{name}信息"),
            )

        self._tokenizer = tokenizer

        # ---- Char-level masks (always built) ----
        self._char_masks: dict[str, str] = {}
        self._build_char_masks()

        # ---- Token-level masks (built only if tokenizer provided) ----
        self._token_masks: dict[str, str] = {}
        self._real_token_ids: dict[str, list[int]] = {}
        self._mask_token_ids: dict[str, list[int]] = {}
        self._palettes = MaskPalettes()

        if tokenizer is not None:
            self._find_palettes()
            self._build_token_masks()

    # ── Read-only properties ────────────────────────────────────────────

    @property
    def field_names(self) -> list[str]:
        """Ordered list of field names (insertion order, Python 3.7+)."""
        return list(self._fields.keys())

    @property
    def palettes(self) -> MaskPalettes:
        return self._palettes

    # ── Per-field accessors ─────────────────────────────────────────────

    def real(self, name: str) -> str:
        """Get the real (unmasked) value of a field."""
        return self._fields[name].real_value

    def char_mask(self, name: str) -> str:
        """Get the character-level mask string (same length as real)."""
        return self._char_masks[name]

    def token_mask(self, name: str) -> str:
        """Get the token-level mask string. Falls back to char mask."""
        return self._token_masks.get(name, self._char_masks[name])

    def real_ids(self, name: str) -> list[int]:
        """Token IDs of the real value."""
        return self._real_token_ids.get(name, [])

    def mask_ids(self, name: str) -> list[int]:
        """Token IDs of the token-level mask."""
        return self._mask_token_ids.get(name, [])

    def description(self, name: str) -> str:
        """Human-readable description of a field."""
        return self._fields[name].description

    def has_tokenizer(self) -> bool:
        return self._tokenizer is not None

    # ── Convenience helpers ─────────────────────────────────────────────

    def get_char_replacement_map(self) -> dict:
        """Return {name: {real:, mask:, description:}} for Method 1 string replacement."""
        return {
            name: {
                "real": self.real(name),
                "mask": self.char_mask(name),
                "description": self.description(name),
            }
            for name in self.field_names
        }

    def replace_sensitive_in_text(self, text: str) -> str:
        """Replace all real values with their char masks in the given text."""
        result = text
        for name in self.field_names:
            result = result.replace(self.real(name), self.char_mask(name))
        return result

    def has_any_real_value(self, text: str) -> bool:
        """Check if text contains any real sensitive value."""
        for name in self.field_names:
            if self.real(name) in text:
                return True
        return False

    # ── Internal: char-level masks ──────────────────────────────────────

    def _build_char_masks(self) -> None:
        """Build same-length char masks: keep '-' / '@' / '.', replace else with '*'."""
        for name, field in self._fields.items():
            chars = []
            for ch in field.real_value:
                if ch in ("-", "@", "."):
                    chars.append(ch)
                else:
                    chars.append("*")
            self._char_masks[name] = "".join(chars)

    # ── Internal: token-level masks ─────────────────────────────────────

    def _find_palettes(self) -> None:
        """Find Unicode block chars that encode to exactly 1 token.

        Ported from TokenLevelMasker._find_single_token_chars / _find_single_token_space_chars.
        """
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
        # Regular single-token chars
        for ch in candidates:
            ids = self._tokenizer.encode(ch, add_special_tokens=False)
            if len(ids) == 1:
                self._palettes.regular.append(ch)
                if len(self._palettes.regular) >= 20:
                    break
        if not self._palettes.regular:
            self._palettes.regular = ["_"]

        # Space-prefixed single-token chars
        for ch in self._palettes.regular:
            ids = self._tokenizer.encode(" " + ch, add_special_tokens=False)
            if len(ids) == 1:
                self._palettes.space_prefixed.append(ch)

    def _build_token_masks(self) -> None:
        """For each field, build a mask string with the same token count as the real value.

        Ported from TokenLevelMasker.mask_value().
        """
        palette = self._palettes.regular

        for name, field in self._fields.items():
            real_ids = self._tokenizer.encode(
                field.real_value, add_special_tokens=False
            )
            self._real_token_ids[name] = real_ids
            target_len = len(real_ids)

            # Build initial mask from alternating palette chars
            parts = []
            for i in range(target_len):
                parts.append(palette[i % len(palette)])
            mask_str = "".join(parts)
            mask_ids = self._tokenizer.encode(mask_str, add_special_tokens=False)

            # Greedily adjust until token count matches
            attempt = 0
            while len(mask_ids) != target_len and attempt < 20:
                attempt += 1
                if len(mask_ids) < target_len:
                    mask_str += palette[attempt % len(palette)]
                else:
                    mask_str = mask_str[:-1]
                mask_ids = self._tokenizer.encode(mask_str, add_special_tokens=False)

            if len(mask_ids) != target_len:
                # Fallback: use '*' repeated (may not be token-aligned, but won't crash)
                mask_str = "*" * len(field.real_value)
                mask_ids = self._tokenizer.encode(mask_str, add_special_tokens=False)

            self._token_masks[name] = mask_str
            self._mask_token_ids[name] = mask_ids
