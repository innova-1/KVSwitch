"""Demo launcher for Baseline 1: All Secrets Visible."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from core.sensitive_config import SensitiveRegistry
from methods.baseline1_visible import Baseline1Visible
from examples.example_config import EXAMPLES, load_example, list_examples

DEFAULT_MODEL = "/data2/models/Qwen/Qwen3-8B"


def run_demo(model_name=DEFAULT_MODEL, device="auto", example_name=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples_to_run = [load_example(example_name)] if example_name else list_examples()
    if not examples_to_run:
        print("没有可用的示例。")
        return

    for ex in examples_to_run:
        print(f"\n{'=' * 60}")
        print(f"示例: {ex['name']} — {ex['description']}")
        print(f"{'=' * 60}")

        registry = SensitiveRegistry(
            fields=ex["sensitive_fields"],
            descriptions=ex["field_descriptions"],
            tokenizer=tokenizer,
        )
        bot = Baseline1Visible(registry, model_name=model_name, device=device)
        bot.run_repl(initial_task=ex["task_prompt"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--example", default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for ex in list_examples():
            print(f"  {ex['module']:20s} — {ex['name']}: {ex['description']}")
    else:
        run_demo(args.model, args.device, args.example)
