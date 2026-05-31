"""
Demo launcher for Method 2: KV-Cache Prefix Splice — multi-turn REPL.

Usage:
    python demos/method2_demo.py [--model /path/to/model] [--device cuda|cpu|auto]
    python demos/method2_demo.py --example email_and_sms
    python demos/method2_demo.py --list

Requirements:
    - HuggingFace transformers
    - A local instruction-tuned causal LM (default: Qwen3-8B)
    - GPU recommended (KV variants consume ~500MB each at 512-token seq len)

Examples are configured in examples/example_config.py.
"""

import sys
import os
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from core.sensitive_config import SensitiveRegistry
from methods.method2_chat import KVCacheChatBot
from examples.example_config import EXAMPLES, load_example, list_examples

# Default model path — edit or override with --model
DEFAULT_MODEL = "/data2/models/Qwen/Qwen3-8B"


def run_demo(model_name=DEFAULT_MODEL, device="auto", example_name=None):
    """Run Method 2 demo with examples from config.

    Args:
        model_name: Path to HuggingFace model.
        device: "auto", "cuda", or "cpu".
        example_name: Specific example to run (None = all enabled).
    """
    # ── Load tokenizer first (needed for token-level masks) ─────────
    print(f"加载 tokenizer: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if example_name:
        examples_to_run = [load_example(example_name)]
    else:
        examples_to_run = list_examples()

    if not examples_to_run:
        print("没有可用的示例。请检查 examples/example_config.py 中的 EXAMPLES 列表。")
        return

    for i, ex in enumerate(examples_to_run):
        print(f"\n{'=' * 60}")
        print(f"示例 {i + 1}/{len(examples_to_run)}: {ex['name']}")
        print(f"描述: {ex['description']}")
        print(f"{'=' * 60}")

        registry = SensitiveRegistry(
            fields=ex["sensitive_fields"],
            descriptions=ex["field_descriptions"],
            tokenizer=tokenizer,
        )

        bot = KVCacheChatBot(registry, model_name=model_name, device=device)
        bot.run_repl(initial_task=ex["task_prompt"])

        if i < len(examples_to_run) - 1:
            input("\n继续下一个示例？按 Enter...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Method 2: KV-Cache Prefix Splice Multi-Turn Demo"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Path to HuggingFace model",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--example", type=str, default=None,
        help="Run a specific example (module name, e.g. email_and_sms)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available examples and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("可用的示例:")
        for ex in list_examples():
            print(f"  {ex['module']:20s} — {ex['name']}: {ex['description']}")
        print(f"\n当前启用的示例列表 (examples/example_config.py):")
        for name in EXAMPLES:
            print(f"  - {name}")
        sys.exit(0)

    run_demo(
        model_name=args.model,
        device=args.device,
        example_name=args.example,
    )
