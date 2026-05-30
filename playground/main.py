#!/usr/bin/env python3
"""
Sensitive-Info-Tracking Demo
============================

Two methods for tracking and protecting sensitive information during LLM
conversations:

  Method 1 — DeepSeek API (Context Rewriting)
    Uses the DeepSeek API with a proxy that intercepts special tokens,
    injects sensitive info on demand, and replaces it with same-length
    masks after task completion.

  Method 2 — Local Model (KV-Cache Manipulation)
    Pre-loads masked sensitive info into the initial context. When the
    model needs a field, the KV-cache is selectively updated. After the
    task, the original masked KV-cache is restored — the model state
    reverts as if it never saw the sensitive data.

Usage:
    python main.py              # Interactive menu
    python main.py --method 1   # Run Method 1 directly
    python main.py --method 2   # Run Method 2 directly
"""

import argparse
import sys

from colorama import Fore, Style

from utils import print_header


def main():
    parser = argparse.ArgumentParser(
        description="Sensitive-Info-Tracking Demo"
    )
    parser.add_argument(
        "--method", type=int, choices=[1, 2],
        help="Which method to run (1=DeepSeek API, 2=KV-Cache)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name for Method 2 (default: Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--api-key", type=str, default="sk-32276abe79c74fff96cf2e3672e8f1ca",
        help="DeepSeek API key for Method 1"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for Method 2 (auto/cpu/cuda)"
    )
    args = parser.parse_args()

    method = args.method
    if method is None:
        method = _show_menu()

    if method == 1:
        _run_method1(args.api_key)
    elif method == 2:
        _run_method2(args.model, args.device)


def _show_menu() -> int:
    print_header("敏感信息追踪 Demo")
    print(f"  {Fore.GREEN}1{Style.RESET_ALL}. 方法一：DeepSeek API — 上下文重写")
    print(f"      通过 API 代理检测特殊 token，注入/替换敏感信息")
    print()
    print(f"  {Fore.GREEN}2{Style.RESET_ALL}. 方法二：本地模型 KV-Cache — Cache 操纵")
    print(f"      将敏感信息预置在上下文开头，通过操纵 KV-Cache 实现加解密")
    print()
    try:
        choice = input(f"{Fore.YELLOW}请选择 (1/2): {Style.RESET_ALL}").strip()
        if choice in ("1", "2"):
            return int(choice)
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    print(f"{Fore.RED}无效选择，默认运行方法二。{Style.RESET_ALL}")
    return 2


def _run_method1(api_key):
    from method1_deepseek import run_method1
    try:
        run_method1(api_key=api_key)
    except ValueError as e:
        print(f"{Fore.RED}错误: {e}{Style.RESET_ALL}")
        print("请设置环境变量 DEEPSEEK_API_KEY 或通过 --api-key 参数传入。")
        sys.exit(1)


def _run_method2(model, device):
    from method2_kvcache import run_method2
    from config import LOCAL_MODEL_NAME
    model_name = model or LOCAL_MODEL_NAME
    print(f"{Fore.BLUE}使用模型: {model_name}{Style.RESET_ALL}")
    try:
        run_method2(model_name=model_name, device=device)
    except Exception as e:
        print(f"{Fore.RED}错误: {e}{Style.RESET_ALL}")
        print("如果因为网络问题无法下载模型，可以尝试使用更小的模型：")
        print("  python main.py --method 2 --model gpt2")
        print("或设置 HF_MIRROR 环境变量使用镜像站。")
        raise


if __name__ == "__main__":
    main()
