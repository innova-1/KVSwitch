# KV-Cache 敏感信息追踪 Demo

五种 LLM 敏感信息保护策略的对比演示。

## 核心问题

LLM 在执行任务时可能需要使用敏感信息（API 密钥、手机号、邮箱等），但用户不希望这些信息残留在模型上下文或 KV-Cache 中。本项目对比了五种不同层级的保护策略：从完全不保护到 KV-Cache 张量级清除 + 长度对齐。

## 项目结构

```
kvcache_3/
├── core/                          # 核心模块
│   ├── sensitive_config.py        # SensitiveRegistry — 运行时定义字段和掩码
│   ├── prompt_builder.py          # PromptBuilder — 动态生成提示词、JSON action 格式
│   └── kv_ops.py                  # KV-Cache 操作 + KVVariantBuilder
├── methods/                       # 五种方法
│   ├── method1_chat.py            # M1: DeepSeek API 上下文重写
│   ├── method2_chat.py            # M2: KV-Cache Prefix Splice
│   ├── method3_inject.py          # M3: 文本注入 + KV 替换 + 长度保持 Padding
│   ├── baseline1_visible.py       # B1: 所有密文直接可见
│   └── baseline2_hidden.py        # B2: 密文始终隐藏（占位符 + 后处理）
├── demos/                         # 启动器
│   ├── method1_demo.py
│   ├── method2_demo.py
│   ├── method3_demo.py
│   ├── baseline1_demo.py
│   └── baseline2_demo.py
├── examples/                      # 对话例子（可配置）
│   ├── example_config.py          # 注册表 — 增删例子只需修改 EXAMPLES
│   ├── email_and_sms.py           # 邮件+短信
│   ├── multi_round.py             # 多轮任务
│   ├── short_values_long_context.py  # 短密文+长上下文（测试对齐）
│   ├── long_values.py             # 长密文（64字符密钥/JWT/URL）
│   └── custom_template.py         # 自定义例子模板
├── playground/                    # 原始代码（不动）
├── utils.py                       # 共享打印工具
└── requirements.txt
```

## 输出格式

所有方法统一使用 JSON Function-Calling 风格：

```json
{"action": "send_email", "params": {"to": "user@example.com", "subject": "会议通知", "body": "..."}}
{"action": "send_sms", "params": {"to": "13800138000", "body": "请查收邮件。"}}
{"action": "task_complete", "params": {}}
<<SENSITIVE_CLEAR>>
```

每行一个 JSON 对象，`task_complete` 后紧跟 `<<SENSITIVE_CLEAR>>`（B1/B2 除外）。

## 快速开始

```bash
pip install -r requirements.txt

# 列出可用例子
python demos/method1_demo.py --list

# M1: DeepSeek API
export DEEPSEEK_API_KEY=sk-xxx
python demos/method1_demo.py --example email_and_sms

# M2/M3/B1/B2: 本地模型（需 GPU）
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

创建新例子：复制 `examples/custom_template.py` → 重命名 → 填写字段 → 加入 EXAMPLES。

---

## 五种方法概览

| | M1 | M2 | M3 | B1 | B2 |
|---|---|---|---|---|---|
| **模型** | DeepSeek API | 本地 Qwen3-8B | 本地 Qwen3-8B | 本地 Qwen3-8B | 本地 Qwen3-8B |
| **密文注入** | API 消息注入 | KV prefix splice | 用户 turn 文本注入 | 始终在 prompt 中 | `<<FIELD:name>>` 占位符 |
| **清除方式** | 字符串 `***` 替换 | splice 回全保密 + remove + 摘要 | remove + 摘要 + pad | — | — |
| **摘要生成** | — | 预测性（看不到完成结果） | 预测性（看不到完成结果） | — | — |
| **敏感段保存** | — | ✓ 可切换回 | ✓ 可切换回 | — | — |
| **长度对齐** | — | ✓ padding | ✓ padding | — | — |
| **KV 变体** | N/A | 2^N 预计算 | 1 个 base KV | 1 个（全真实） | 1 个（无密文） |
| **多轮对话** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **quit 审计** | ✓ 消息历史 | ✓ KV 对话日志 | ✓ KV 对话日志 | — | — |
| **外泄风险** | 有（经过 API） | 无 | 无 | 有（始终可见） | 无 |

---

### Method 1: DeepSeek API — 上下文重写

消息/文本层面操作。`<<SENSITIVE_REQUEST:field>>` 请求 → 注入真实值 → `<<SENSITIVE_CLEAR>>` → 全文 `***` 掩码替换。quit 后打印消息历史审计。

```
request → inject text → use → clear (string replace) → audit
```

### Method 2: KV-Cache Prefix Splice

KV-Cache 张量层面。敏感信息在 prefix 中，预计算 2^N 个 variant。`splice_prefix` 交换 prefix；Phase 3 用 `clean_position` 边界删除污染后缀 → 从干净上下文生成预测性摘要 → 敏感段保存 → padding 对齐长度。

```
Phase 0: prefill 2^N variants
Phase 1: masked generate → request
Phase 2: splice prefix → hint → generate with real values
Phase 3: save segment → delete polluted suffix → predictive summary → pad → re-prefill
```

### Method 3: 文本注入 + KV 替换 + 长度保持

离线版 M1。密文以用户 turn 文本注入（非 prefix splice）。`clean_position` 标记干净边界 → 删除污染后缀 → 预测性摘要 → padding 保持位置编码 → 敏感段保存以备切换回。

```
request → inject text → use → clear: save segment + delete + summary + pad
```

### Baseline 1: 所有密文直接可见

敏感值直接写在系统提示词中，模型全程可见。最简基线。

```
prompt with real values → generate → return
```

### Baseline 2: 密文始终隐藏

模型输入不含真实值，输出 `<<FIELD:email>>` 占位符。后处理器仅替换 **JSON action 行** 中的占位符为真实值。模型永远看不到密文。

```
masked prompt → generate with <<FIELD:name>> → post-process JSON lines only → return
```

---

## 模块依赖

```
core/sensitive_config.py
     │
     ├── core/prompt_builder.py
     │        │
     │        ├── methods/method1_chat.py ── demos/method1_demo.py
     │        ├── methods/method2_chat.py ── demos/method2_demo.py  ← core/kv_ops.py
     │        ├── methods/method3_inject.py ── demos/method3_demo.py ← core/kv_ops.py
     │        ├── methods/baseline1_visible.py ── demos/baseline1_demo.py
     │        └── methods/baseline2_hidden.py ── demos/baseline2_demo.py
     │
     └── utils.py
```
