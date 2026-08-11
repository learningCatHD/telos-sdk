<div align="center">

<a href="https://github.com/eryflow/telos-sdk">
  <img src="https://img.shields.io/badge/NEW_HOME-eryflow%2Ftelos--sdk-0969DA?style=for-the-badge&logo=github&logoColor=white" alt="新仓库：eryflow/telos-sdk" />
</a>

## TELOS SDK 已迁移至新家

本项目已迁移至 **[eryflow/telos-sdk](https://github.com/eryflow/telos-sdk)**。

[**前往新仓库 →**](https://github.com/eryflow/telos-sdk)

</div>

---

<div align="center">

<img src="assets/logo.svg" alt="TELOS — 可移植 Agent 上下文" width="460"/>

### 上下文归你所有 &nbsp;·&nbsp; Agent 是雇来的

**无需重写。无需压缩。可节省高达 90% token 账单。**

<sub>💰 **token 账单 −50–90%** &nbsp;·&nbsp; 🎯 **agent 行为不变** &nbsp;·&nbsp; ⚡ **更快，不更慢** &nbsp;·&nbsp; 🔒 **不捕获任何内容**</sub>

<sub>一份唯一 IR——tools、system、turns 与 memory——可在 Anthropic · OpenAI · DeepSeek · vLLM · SGLang 上不加修改地运行</sub>

<sub>清华大学 LEAP Lab —— 聚焦机器学习、多模态学习与具身智能的研究团队 · <a href="https://www.leaplab.ai/">leaplab.ai</a></sub>

<br/>

[![Core](https://img.shields.io/badge/core-Apache%202.0-2C5F66?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-4FB3BF?style=flat-square)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-Beta-d8851f?style=flat-square)](CHANGELOG.md)
[![Protocol](https://img.shields.io/badge/protocol-TELOS%20IR-7FD8E0?style=flat-square)](https://docs.telosai.pro/zh/concepts/protocol)

### 📖 完整文档 → **[docs.telosai.pro](https://docs.telosai.pro/zh)**

[**快速开始**](#quickstart) &nbsp;·&nbsp; [**四个承诺**](#guarantees) &nbsp;·&nbsp; [**文档**](https://docs.telosai.pro/zh) &nbsp;·&nbsp; [**Benchmark**](https://docs.telosai.pro/zh/benchmark/swebench) &nbsp;·&nbsp; [**协议**](https://docs.telosai.pro/zh/concepts/protocol)

[📖 English](README.md) &nbsp;|&nbsp; **🇨🇳 简体中文**

</div>

---

**最新动态** 🔥

* **[2026.06.06]** 文档站正式上线 → **[docs.telosai.pro](https://docs.telosai.pro/zh)** —— 完整指南、协议详解、支持矩阵与 SWE-bench 报告，中英文双语。
* **[2026.05.31]** 与 [cc-switch](https://github.com/farion1231/cc-switch) 共存 —— TELOS 把网关挡在 cc-switch 选定的上游中转前面，不会有任何密钥被写入 TELOS 配置。
* **[2026.05.29]** `telos init` 现在会在注册新的 harness 上游时自动重启网关，省去手动重启那一步。
* **[2026.05.27]** Codex.app（ChatGPT 登录模式）成为一等 harness；安装器自动检测 `auth_mode` 并路由到正确的上游。

---

## ⬢ &nbsp;TELOS 是什么？

TELOS 是一个挡在 agent 与模型之间的缓存感知网关。它重排 proxy→上游这一段，让共享前缀由缓存（`cache_read`）命中，而不是每轮按全价重新计费——**不改你的 prompt、不换你的模型、不改 agent 的行为**。

把一段真实 **6 轮** 会话丢进 openclaw，只改两个开关：

| 模式 | raw input tokens | cache_read | 6 轮总成本 |
|---|:--:|:--:|:--:|
| passthrough（今天的默认） | 24,151 | 0 | **$0.3623** |
| 使用 TELOS | 0 | 18,701 | **$0.0281（−92.3%）** |

放大到 1,000 个会话：**$362 → $26**，每个月都能看见，再乘上团队规模。我们按绝对 $/已解决请求 记录节省——比例可以造，美元不行。

→ 完整背景请见[**文档**](https://docs.telosai.pro/zh)。

<a id="quickstart"></a>

## ⬢ &nbsp;快速开始 —— 3 步省下 90%

```bash
# ❶ 安装 —— 一行脚本（Linux / macOS / WSL2 / Android Termux）
curl -fsSL https://raw.githubusercontent.com/learningCatHD/telos-sdk/main/scripts/install.sh | bash
# …或用 pip：  uv pip install -U telos-sdk

# ❷ 连接 —— 自动检测 claude-code / codex / openclaw / hermes，注入配置，
#    并在后台启动本地网关。不需要改 agent 代码。
telos init

# ❸ 观察 —— 打开离线 HTML 看板，以绝对美元展示节省
telos dashboard
```

<p align="center">
  <img src="assets/05-dashboard.png" alt="TELOS savings dashboard — absolute dollars broken down by harness / model / session" width="100%"/>
</p>

→ 详细的[安装](https://docs.telosai.pro/zh/start/installation)与[快速开始](https://docs.telosai.pro/zh/start/quickstart)指南，以及 [cc-switch 共存](https://docs.telosai.pro/zh/guides/integration-paths)说明，都在文档里。

<a id="guarantees"></a>

## ⬢ &nbsp;你真正关心的四件事

TELOS 改变的是**你被计费的内容**，而不是**你的 agent 做什么**。

| 你关心的 | 承诺 |
|---|---|
| 💰 **Token 账单** | **计费输入 token 降低 50%–90%。** 6 轮真实会话 −92.3%；SWE-bench Verified new_input −52.8% / 端到端成本 −40.5%。 |
| 🎯 **Agent 行为** | **完全不变。** 同一个模型、同样的 prompt 语义、同样的输出。SWE-bench A/B：McNemar p = 0.66，解决率无回归。 |
| ⚡ **推理速度** | **不会更慢，只会更快。** 缓存命中跳过对已提交字节的重新 prefill，会话越长，首 token 时延越低。 |
| 🔒 **你的数据** | **不捕获任何具体内容。** 网关跑在 `127.0.0.1`；用量日志只记录 token 计数，从不记录 prompt/回复正文。无云端、无遥测。 |

→ 每个承诺为何成立的详细论证：[**docs.telosai.pro**](https://docs.telosai.pro/zh)。

## ⬢ &nbsp;了解更多

| 主题 | 位置 |
|---|---|
| **协议** —— 三色带（PIN/FOLD/DROP）与单调追加 | [concepts/protocol](https://docs.telosai.pro/zh/concepts/protocol) · [concepts/bands](https://docs.telosai.pro/zh/concepts/bands) |
| **支持矩阵** —— harness、frontier model、推理框架、cc-switch | [reference/support-matrix](https://docs.telosai.pro/zh/reference/support-matrix) |
| **SWE-bench Verified A/B** —— 预先登记设计、统计细节、完整报告 | [benchmark/swebench](https://docs.telosai.pro/zh/benchmark/swebench) |
| **架构与接入方式** | [concepts/architecture](https://docs.telosai.pro/zh/concepts/architecture) · [guides/integration-paths](https://docs.telosai.pro/zh/guides/integration-paths) |
| **CLI 参考与更新日志** | [reference/cli](https://docs.telosai.pro/zh/reference/cli) · [CHANGELOG.md](CHANGELOG.md) |

**TELOS 是开源的。把它接到你的真实工作流里，看看那 92% 到底是真收益，还是又一个“X 倍 token”说法。**

<a id="citation"></a>

## Citation

Core contributors: Zheng Wang, Shenzhi Wang, HongTao Zhong, Shiji Song, Gao Huang

```bibtex
@misc{wang2026telos-agent,
  title        = {Telos: A Cost-Aware Inference Infrastructure for AI Agent},
  author       = {Zheng Wang, Shenzhi Wang, HongTao Zhong, Shiji Song, Gao Huang},
  howpublished = {\url{https://github.com/learningCatHD/telos-sdk.git}},
  year         = {2026}
}
```

---

<div align="center">
<a href="https://github.com/learningCatHD/telos-sdk"><img src="https://img.shields.io/badge/⭐%20Star%20on%20GitHub-learningCatHD%2Ftelos--sdk-1F4A50?style=for-the-badge&logo=github&logoColor=white" alt="Star on GitHub"/></a>

<sub>📖 完整文档见 <a href="https://docs.telosai.pro/zh">docs.telosai.pro</a></sub>
</div>
