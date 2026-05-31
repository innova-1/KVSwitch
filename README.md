# KV-Cache 敏感信息追踪 Demo

两种在 LLM 推理过程中保护敏感信息的方法对比演示。

## 核心问题

LLM 在处理任务时可能需要使用敏感信息（API 密钥、手机号、邮箱等），但我们不希望这些信息在对话结束后残留于上下文或模型状态中。示例任务里敏感信息默认以掩码形式呈现，真实值仅在模型请求后注入。

## 项目结构

```
kvcache_3/
├── core/                       # 核心模块
│   ├── sensitive_config.py     # SensitiveRegistry — 运行时定义敏感字段和掩码
│   ├── prompt_builder.py       # PromptBuilder — 字段名参数化的提示词生成
│   └── kv_ops.py               # KV-Cache 操作 + KVVariantBuilder
├── methods/                    # 两种方法实现
│   ├── method1_chat.py         # DeepSeekChatBot — 上下文重写 + 多轮 REPL
│   └── method2_chat.py         # KVCacheChatBot — KV Prefix Splice + 多轮 REPL
├── demos/                      # Demo 启动器
│   ├── method1_demo.py         # 方法一启动器（从 examples/ 读取任务）
│   └── method2_demo.py         # 方法二启动器（从 examples/ 读取任务）
├── examples/                   # 对话例子（可配置）
│   ├── example_config.py       # 例子注册表 — 增删例子只需修改 EXAMPLES 列表
│   ├── email_and_sms.py        # 邮件+短信发送示例
│   ├── multi_round.py          # 多轮任务示例
│   └── custom_template.py      # 自定义例子模板
├── playground/                 # 原始代码（参考用，保持不动）
│   ├── config.py, utils.py
│   ├── method1_deepseek.py, method2_kvcache.py
│   └── main.py
├── utils.py                    # 共享工具（彩色输出）
└── requirements.txt
```

## 输出格式（JSON Function-Calling 风格）

两种方法统一使用结构化 JSON 输出操作：

```json
{"action": "send_email", "params": {"to": "user@example.com", "subject": "会议通知", "body": "会议改到明天下午3点。"}}
{"action": "send_sms", "params": {"to": "13800138000", "body": "请查收邮件。"}}
{"action": "task_complete", "params": {}}
```

每行一个 JSON 对象，可被程序直接解析。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 列出可用例子
python demos/method1_demo.py --list
python demos/method2_demo.py --list

# 方法一：运行所有启用的例子
export DEEPSEEK_API_KEY=sk-xxx
python demos/method1_demo.py

# 方法一：运行特定例子
python demos/method1_demo.py --example email_and_sms

# 方法二：运行所有启用的例子（需 GPU / 本地模型）
python demos/method2_demo.py --model /path/to/Qwen3-8B

# 方法二：运行特定例子 + CPU 模式
python demos/method2_demo.py --example email_and_sms --device cpu
```

## 添加/删除对话例子

编辑 `examples/example_config.py`：

```python
EXAMPLES = [
    "email_and_sms",
    "multi_round",      # ← 取消注释即可启用
    # "my_new_task",    # ← 在这里添加你新写的例子
]
```

创建新例子：复制 `examples/custom_template.py` → 重命名 → 填写字段 → 在 `example_config.py` 的 `EXAMPLES` 中加入模块名。

## 模块设计

```
sensitive_config.py (LEAF)
         │
         ├── prompt_builder.py
         │        │
         │        ├── method1_chat.py ── method1_demo.py
         │        │
         │        └── kv_ops.py ── method2_chat.py ── method2_demo.py
         │
         └── utils.py (shared print helpers)
```

- `SensitiveRegistry(fields={...})` — 运行时定义敏感字段，自动生成字符级/Token级掩码
- `PromptBuilder(registry)` — 所有提示词从字段名动态生成，无硬编码
- `KVVariantBuilder` — 为任意字段集构建 2^N 个 KV-Cache 变体
- 审计功能（quit 后）：自动扫描历史消息，验证敏感信息是否已清除

## 方法一：DeepSeek API — 上下文重写

在消息/文本层面操作。AI 通过 `<<SENSITIVE_REQUEST:field>>` 请求字段，Proxy 注入真实值；AI 输出 `<<SENSITIVE_CLEAR>>` 时，Proxy 全文扫描替换为等长掩码。

## 方法二：本地模型 — KV-Cache Prefix Splice

在 KV-Cache 张量层面操作。敏感信息放在输入最开头，预计算 2^N 个 KV 变体，通过 `splice_prefix` 拼接交换 prefix 部分。Phase 3 用 `remove_token_range` 切除敏感输出 token 并用摘要替换。

## 技术对比

| | 方法一 | 方法二 |
|---|---|---|
| 掩码方式 | 字符级等长 `*` | Token 级对齐（Unicode block chars） |
| 敏感信息位置 | 消息列表中 | 提示词最开头（prefix） |
| 加解密操作 | 字符串替换 | Tensor splice |
| 清除方式 | 全文扫描替换 | splice 回全保密 prefix + remove_token_range |
| 对外依赖 | DeepSeek API | 本地模型 |
| 敏感信息外泄风险 | 有（经过 API 服务器） | 无 |
