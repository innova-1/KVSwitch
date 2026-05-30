# KV-Cache 敏感信息追踪 Demo

两种在 LLM 推理过程中保护敏感信息的方法对比演示。

## 核心问题

LLM 在处理任务时可能需要使用敏感信息（API 密钥、手机号、邮箱等），但我们不希望这些信息在对话结束后残留于上下文或模型状态中。示例任务里邮箱/手机号默认以掩码形式呈现，真实值仅在模型请求后注入。

## 方法一：DeepSeek API — 上下文重写

```
用户任务 → DeepSeek → AI 输出 <<SENSITIVE_REQUEST:api_key>>
                                ↓
                       Proxy 注入真实值到消息列表
                                ↓
                      AI 使用真实值完成任务 → 输出 <<SENSITIVE_CLEAR>>
                                ↓
                       Proxy 扫描全部消息，将真实值替换为等长掩码
                                ↓
                      继续正常对话（上下文已脱敏）
```

**关键机制**：在消息层面做字符串替换，真实值替换为相同长度的 `***` 掩码。

**优势**：实现简单，不依赖模型内部状态，适用于任何 API。

**劣势**：敏感信息在替换前已存在于 API 提供方的服务器上。

---

## 方法二：本地模型 — KV-Cache Prefix Splice

```
                    Phase 0: 预计算 8 个 KV
                    ═══════════════════════
   全保密 prompt ──prefill──→ kv_all
   A公开 prompt  ──prefill──→ kv_api
   B公开 prompt  ──prefill──→ kv_phone
   C公开 prompt  ──prefill──→ kv_email
   AB公开 prompt ──prefill──→ kv_api_phone
   AC公开 prompt ──prefill──→ kv_api_email
   BC公开 prompt ──prefill──→ kv_phone_email
   ABC公开 prompt ──prefill──→ kv_all_three

                    Phase 1: 全保密推理
                    ════════════════════
   kv_all ──生成──→ "... <<SENSITIVE_REQUEST:api_key>>"

                    Phase 2: Splice Prefix
                    ═════════════════════
   splice_prefix(kv_api, current_kv, prefix_len)
   ┌──────────────────────┬────────────────────────┐
   │ kv_api 的 prefix     │ current_kv 的 suffix   │
   │ (api_key 明文)       │ (不变！)               │
   │ tokens [0..P)        │ tokens [P..current)    │
   └──────────────────────┴────────────────────────┘

                    Phase 3: 恢复全保密
                    ════════════════════
   splice_prefix(kv_all, current_kv, prefix_len)
   → prefix 恢复掩码状态，suffix 不变
   → 敏感信息从模型状态完全消失
```

**关键机制**：在 KV-cache 张量层面做 prefix 拼接，仅替换最前面敏感信息对应的 KV，其余部分原样保留。

**优势**：敏感信息永远不离开本地，清除后模型状态中无痕迹。

**劣势**：需要本地运行模型，需要 token 级别对齐保证 KV shape 一致。

---

## 项目结构

```
kvcache/
├── playground/
│   ├── config.py              # 共享配置：special tokens、敏感信息定义
│   ├── utils.py               # 共享工具：彩色输出、prompt 构建
│   ├── method1_deepseek.py    # 方法一：DeepSeek API 上下文重写
│   ├── method2_kvcache.py     # 方法二：本地模型 KV-Cache Prefix Splice
│   ├── main.py                # 入口
│   └── deepseek_api.env.sh    # API Key 模板（本地使用，已在 .gitignore）
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 进入 playground
cd playground

# 交互式菜单
python main.py

# 运行方法一（需要 DeepSeek API Key）
source ./deepseek_api.env.sh
python main.py --method 1

# 运行方法二（自动下载 Qwen2.5-0.5B-Instruct，约 1GB）
python main.py --method 2

# 方法二使用更小的模型
python main.py --method 2 --model gpt2

# 方法二指定 CPU
python main.py --method 2 --device cpu
```

## DeepSeek API Key（避免误提交）

- 编辑 [kvcache/playground/deepseek_api.env.sh](kvcache/playground/deepseek_api.env.sh) 填入真实 Key。
- 该文件已在 [kvcache/.gitignore](kvcache/.gitignore) 中忽略，避免提交到 Git。

## 方法二的核心操作

`splice_prefix` 是整个方法二的精髓（[playground/method2_kvcache.py](playground/method2_kvcache.py#L422-L441)）：

```python
def splice_prefix(source_kv, target_kv, prefix_len):
    """替换 target_kv 前 prefix_len 个位置的 KV 为 source_kv 的对应部分"""
    for layer_s, layer_t in zip(source_kv, target_kv):
        key_s, val_s = layer_s   # source KV
        key_t, val_t = layer_t   # target KV
        new_key = torch.cat([
            key_s[:, :, :prefix_len, :],   # ← source 的 prefix
            key_t[:, :, prefix_len:, :],   # ← target 的 suffix（不变！）
        ], dim=2)
        ...
```

每次 splice 只需要做一次 tensor concatenation，无需重新 forward pass。8 个 KV 变体在初始化时预计算好，后续推理中按需 splice。

## 技术要点

| | 方法一 | 方法二 |
|---|---|---|
| 掩码方式 | 字符级等长 `*` | Token 级对齐（交替 Unicode block chars） |
| 敏感信息位置 | 消息列表中 | 提示词最开头（prefix） |
| 加解密操作 | 字符串替换 | Tensor splice |
| 清除方式 | 全文扫描替换 | splice 回全保密 prefix |
| 对外依赖 | DeepSeek API | 本地模型 |
| 敏感信息外泄风险 | 有（经过 API 服务器） | 无 |
