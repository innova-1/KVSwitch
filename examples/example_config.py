"""
Example registry — controls which dialogue examples are available.

To add a new example:
  1. Create a .py file in this directory (e.g., my_task.py)
  2. Define EXAMPLE_NAME, EXAMPLE_DESC, TASK_PROMPT, SENSITIVE_FIELDS, FIELD_DESCRIPTIONS
  3. Add the module name to EXAMPLES below

To remove an example:
  Comment out or delete its entry in EXAMPLES.

Each example module must export:
  EXAMPLE_NAME: str           — Human-readable name
  EXAMPLE_DESC: str           — One-line description
  TASK_PROMPT: str            — The initial user task prompt
  SENSITIVE_FIELDS: dict      — {field_name: real_value}
  FIELD_DESCRIPTIONS: dict    — Optional {field_name: description}
"""

import importlib

# ── Active examples ────────────────────────────────────────────────────────
# Comment out any line to disable. Add new module names to enable.

EXAMPLES = [
    "email_and_sms",
    # "multi_round",     # uncomment to enable
]


# ── Registry helpers ────────────────────────────────────────────────────────

def load_example(name: str) -> dict:
    """Load an example module and return its config dict.

    Returns a dict with keys:
      name, description, task_prompt, sensitive_fields, field_descriptions
    """
    mod = importlib.import_module(f"examples.{name}")
    return {
        "module": name,
        "name": getattr(mod, "EXAMPLE_NAME", name),
        "description": getattr(mod, "EXAMPLE_DESC", ""),
        "task_prompt": mod.TASK_PROMPT,
        "sensitive_fields": mod.SENSITIVE_FIELDS,
        "field_descriptions": getattr(mod, "FIELD_DESCRIPTIONS", {}),
    }


def list_examples() -> list[dict]:
    """Load all enabled examples from EXAMPLES."""
    results = []
    for name in EXAMPLES:
        try:
            results.append(load_example(name))
        except Exception as e:
            print(f"[WARN] 无法加载示例 '{name}': {e}")
    return results
