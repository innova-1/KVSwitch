"""
Demo launcher for Method 1: DeepSeek API — multi-turn REPL.

Usage:
    python demos/method1_demo.py                    # run all enabled examples
    python demos/method1_demo.py --example email_and_sms  # run a specific example
    python demos/method1_demo.py --list              # list available examples

Requires DEEPSEEK_API_KEY environment variable or edit api_key below.

Examples are configured in examples/example_config.py.
"""

import os
import sys
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sensitive_config import SensitiveRegistry
from methods.method1_chat import DeepSeekChatBot
from examples.example_config import EXAMPLES, load_example, list_examples


DEFAULT_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def run_demo(example_name=None, api_key=None):
    """Run Method 1 demo with examples from config.

    Args:
        example_name: Specific example module name to run (None = all enabled).
        api_key: DeepSeek API key.
    """
    api_key = api_key or DEFAULT_API_KEY
    if not api_key:
        print("请设置 DEEPSEEK_API_KEY 环境变量 或使用 --api-key 参数")
        sys.exit(1)

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
        )

        bot = DeepSeekChatBot(registry, api_key=api_key)
        bot.run_repl(initial_task=ex["task_prompt"])

        if i < len(examples_to_run) - 1:
            input("\n继续下一个示例？按 Enter...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Method 1: DeepSeek API Multi-Turn Demo"
    )
    parser.add_argument(
        "--example", type=str, default=None,
        help="Run a specific example (module name, e.g. email_and_sms)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available examples and exit",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="DeepSeek API key",
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

    run_demo(example_name=args.example, api_key=args.api_key)
