# KV-Cache 敏感信息追踪 Demo

五种在 LLM 推理过程中保护敏感信息的方法对比演示。

## 核心问题

LLM 在处理任务时可能需要使用敏感信息（API 密钥、手机号、邮箱等），但我们不希望这些信息在对话结束后残留于上下文或模型状态中。本项目对比了五种不同层级的保护策略。

## 项目结构

```
kvcache_3/
├── core/                          # 核心模块
│   ├── sensitive_config.py        # SensitiveRegistry — 运行时定义敏感字段和掩码
│   ├── prompt_builder.py          # PromptBuilder — 动态生成提示词、JSON action 格式
│   └── kv_ops.py                  # KV-Cache 操作 + KVVariantBuilder + remove_token_range
├── methods/                       # 五种方法实现
│   ├── method1_chat.py            # Method 1: DeepSeek API 上下文重写
│   ├── method2_chat.py            # Method 2: KV-Cache Prefix Splice
│   ├── method3_inject.py          # Method 3: 文本注入 + KV 替换 + 长度保持 Padding
│   ├── baseline1_visible.py       # Baseline 1: 所有密文直接可见
│   └── baseline2_hidden.py        # Baseline 2: 密文始终隐藏（占位符 + 后处理）
├── demos/                         # Demo 启动器（每个方法一个）
│   ├── method1_demo.py            # Method 1 启动器
│   ├── method2_demo.py            # Method 2 启动器
│   ├── method3_demo.py            # Method 3 启动器
│   ├── baseline1_demo.py          # Baseline 1 启动器
│   └── baseline2_demo.py          # Baseline 2 启动器
├── examples/                      # 对话例子（可配置）
│   ├── example_config.py          # 例子注册表 — 增删例子只需修改 EXAMPLES 列表
│   ├── email_and_sms.py           # 邮件+短信发送示例
│   ├── multi_round.py             # 多轮任务示例
│   ├── short_values_long_context.py  # 短密文+长上下文（测试对齐）
│   ├── long_values.py             # 长密文（64字符密钥/JWT/长URL）
│   └── custom_template.py         # 自定义例子模板
├── playground/                    # 原始代码（参考用，保持不动）
│   ├── config.py, utils.py
│   ├── method1_deepseek.py, method2_kvcache.py
│   └── main.py
├── utils.py                       # 共享工具（彩色输出）
└── requirements.txt
```

## 输出格式（JSON Function-Calling 风格）

所有方法统一使用结构化 JSON 输出操作：

```json
{"action": "send_email", "params": {"to": "user@example.com", "subject": "会议通知", "body": "会议改到明天下午3点。"}}
{"action": "send_sms", "params": {"to": "13800138000", "body": "请查收邮件。"}}
{"action": "task_complete", "params": {}}
<<SENSITIVE_CLEAR>>
```

每行一个 JSON 对象，`task_complete` 后紧跟 `<<SENSITIVE_CLEAR>>`（除 Baseline 外）。

## 快速开始

```bash
pip install -r requirements.txt

# 列出可用例子
python demos/method1_demo.py --list

# Method 1: DeepSeek API（需 API Key）
export DEEPSEEK_API_KEY=sk-xxx
python demos/method1_demo.py --example email_and_sms

# Method 2 + Method 3 + Baseline 1/2: 本地模型（需 GPU）
python demos/method2_demo.py --model /data2/models/Qwen/Qwen3-8B
python demos/method3_demo.py --model /data2/models/Qwen/Qwen3-8B
python demos/baseline1_demo.py --model /data2/models/Qwen/Qwen3-8B
python demos/baseline2_demo.py --model /data2/models/Qwen/Qwen3-8B

# 运行特定例子
python demos/method2_demo.py --example short_values_long_context
```

## 添加/删除对话例子

编辑 `examples/example_config.py`：

```python
EXAMPLES = [
    "email_and_sms",
    # "multi_round",                   # 取消注释启用
    # "short_values_long_context",     # 短密文+长上下文
    # "long_values",                   # 长密文
]
```

创建新例子：复制 `examples/custom_template.py` → 重命名 → 填写字段 → 加入 EXAMPLES 列表。

---

## 五种方法概览

| | Method 1 | Method 2 | Method 3 | Baseline 1 | Baseline 2 |
|---|---|---|---|---|---|
| **模型** | DeepSeek API | 本地 Qwen3-8B | 本地 Qwen3-8B | 本地 Qwen3-8B | 本地 Qwen3-8B |
| **密文注入** | API 消息注入 | KV prefix splice | 用户 turn 文本注入 | 始终可见 | 占位符 `<<FIELD:name>>` |
| **清除方式** | 字符串替换 `***` | remove_token_range + summary | remove_token_range + summary + pad | 无需 | 无需（模型从未见到） |
| **KV 变体** | N/A | 2^N 预计算 | 1 个 | 1 个 | 1 个 |
| **多轮对话** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **审计（quit）** | ✓ 消息历史扫描 | ✓ KV 对话审计 | ✓ KV 对话审计 | — | — |
| **外泄风险** | 有（经过 API） | 无 | 无 | 有（始终可见） | 无 |

---

### Method 1: DeepSeek API — 上下文重写

在消息/文本层面操作。通过 `<<SENSITIVE_REQUEST:field>>` 请求字段 → Proxy 注入真实值 → `<<SENSITIVE_CLEAR>>` 全文替换为 `***` 掩码。quit 后打印完整消息历史并审计。

**流程**：`request → inject → use → clear → audit`

### Method 2: KV-Cache Prefix Splice

在 KV-Cache 张量层面操作。敏感信息放在输入最开头（prefix），预计算 2^N 个 KV 变体，通过 `splice_prefix` 交换 prefix。Phase 3 用 `remove_token_range` 切除敏感输出 + prefill 摘要替换。quit 后审计 KV 对话日志。

**流程**：`Phase 0: prefill variants → Phase 1: masked → Phase 2: splice+reveal → Phase 3: remove+summary`

### Method 3: 文本注入 + KV 替换 + 长度保持 Padding

离线版 Method 1。密文通过用户 turn 文本注入（非 prefix splice），清除时 `remove_token_range` 删除污染段 → 生成摘要 → padding 补回原始 KV 长度以保持位置编码一致。quit 后审计。

**流程**：`request → inject text → use → clear: remove+summary+pad`

### Baseline 1: 所有密文直接可见

最简基线。所有敏感值直接写在系统提示词中，模型全程可见。无保护、无清除、无审计。

**流程**：`prompt with real values → generate → return`

### Baseline 2: 密文始终隐藏（占位符 + 后处理）

模型输入中不包含任何真实值。模型输出 `<<FIELD:email>>` 占位符，后处理器仅替换 **JSON action 行** 中的占位符为真实值（普通文本中的占位符保留不动）。模型永远看不到密文。

**流程**：`masked prompt → generate with <<FIELD:name>> → post-process JSON lines only → return`

---

## 模块设计

```
core/sensitive_config.py (LEAF)
         │
         ├── core/prompt_builder.py
         │        │
         │        ├── methods/method1_chat.py ─── demos/method1_demo.py
         │        │
         │        ├── methods/method2_chat.py ─── demos/method2_demo.py
         │        │        └── core/kv_ops.py
         │        │
         │        ├── methods/method3_inject.py ── demos/method3_demo.py
         │        │        └── core/kv_ops.py
         │        │
         │        ├── methods/baseline1_visible.py ── demos/baseline1_demo.py
         │        │
         │        └── methods/baseline2_hidden.py ── demos/baseline2_demo.py
         │
         └── utils.py (shared print helpers)
```

---

## 技术对比

| 维度 | Method 1 | Method 2 | Method 3 | Baseline 1 | Baseline 2 |
|---|---|---|---|---|---|
| 掩码方式 | 字符 `*` | Token 对齐 Unicode | 无需掩码 | 无 | 占位符 |
| 敏感信息位置 | 消息列表 | Prefix | 用户 turn 文本 | 系统提示词 | 不存在 |
| 加解密 | 字符串替换 | Tensor splice | 文本追加+KV删除 | 无 | 后处理 |
| 清除方式 | 全文扫描替换 | KV remove+summary | KV remove+summary+pad | 无 | 无 |
| GPU 需求 | 无 | 需要 | 需要 | 需要 | 需要 |
| 多轮持久 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 位置编码保持 | — | — | ✓（padding） | — | — |
