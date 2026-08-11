<div align="center">

<a href="https://github.com/eryflow/telos-sdk">
  <img src="https://img.shields.io/badge/NEW_HOME-eryflow%2Ftelos--sdk-0969DA?style=for-the-badge&logo=github&logoColor=white" alt="New home: eryflow/telos-sdk" />
</a>

## TELOS SDK has a new home

This project has moved to **[eryflow/telos-sdk](https://github.com/eryflow/telos-sdk)**.

[**Visit the new repository →**](https://github.com/eryflow/telos-sdk)

</div>

---

<div align="center">

<img src="assets/logo.svg" alt="TELOS — Portable Agent Context" width="460"/>

### Context is yours &nbsp;·&nbsp; Agents are hired

**No rewrite. No compression. Up to 90% token billing saving.**

<sub>💰 **−50–90% token bill** &nbsp;·&nbsp; 🎯 **Same agent behavior** &nbsp;·&nbsp; ⚡ **Faster, not slower** &nbsp;·&nbsp; 🔒 **Captures no content**</sub>

<sub>One canonical IR — tools, system, turns, and memory — runs unchanged across Anthropic · OpenAI · DeepSeek · vLLM · SGLang</sub>

<sub>LEAP Lab @ Tsinghua University — machine learning, multimodal learning, and embodied intelligence · <a href="https://www.leaplab.ai/">leaplab.ai</a></sub>

<br/>

[![Core](https://img.shields.io/badge/core-Apache%202.0-2C5F66?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-4FB3BF?style=flat-square)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-Beta-d8851f?style=flat-square)](CHANGELOG.md)
[![Protocol](https://img.shields.io/badge/protocol-TELOS%20IR-7FD8E0?style=flat-square)](https://docs.telosai.pro/en/concepts/protocol)
[![Version](https://img.shields.io/badge/version-0.1.8-4FB3BF?style=flat-square)](CHANGELOG.md)

### 📖 Full documentation → **[docs.telosai.pro](https://docs.telosai.pro)**

[**Quickstart**](#quickstart) &nbsp;·&nbsp; [**Guarantees**](#guarantees) &nbsp;·&nbsp; [**Docs**](https://docs.telosai.pro) &nbsp;·&nbsp; [**Benchmark**](https://docs.telosai.pro/en/benchmark/swebench) &nbsp;·&nbsp; [**Protocol**](https://docs.telosai.pro/en/concepts/protocol)

**📖 English** &nbsp;|&nbsp; [🇨🇳 中文](README.zh-CN.md)

</div>

---

**News** 🔥

* **[2026.06.06]** Documentation site is live → **[docs.telosai.pro](https://docs.telosai.pro)** — full guides, protocol deep-dive, support matrix, and the SWE-bench report, in English and 中文.
* **[2026.05.31]** Coexists with [cc-switch](https://github.com/farion1231/cc-switch) — TELOS chains its gateway in front of whatever upstream relay cc-switch selects, no secret ever copied into TELOS config.
* **[2026.05.29]** `telos init` now auto-restarts the gateway when a new harness upstream is registered, dropping the manual restart step.
* **[2026.05.27]** Codex.app (ChatGPT login mode) is now a first-class harness; the installer auto-detects `auth_mode` and routes through the correct upstream.

---

## ⬢ &nbsp;What is TELOS?

TELOS is a cache-aware gateway that sits between your agent and the model. It restructures the proxy→upstream segment so the shared prefix is served from cache (`cache_read`) instead of being re-billed at full price every turn — **without changing your prompts, your model, or your agent's behavior**.

Take a real **6-turn** conversation through openclaw and flip two switches:

| Mode | raw input tokens | cache_read | Cost for 6 turns |
|---|:--:|:--:|:--:|
| passthrough (today's default) | 24,151 | 0 | **$0.3623** |
| with TELOS | 0 | 18,701 | **$0.0281 (−92.3%)** |

Scale to 1,000 sessions: **$362 → $26**, every month, multiplied by team size. We report savings in absolute $/query-resolved — ratios can be gamed; dollars can't.

→ Read the full story in the [**docs**](https://docs.telosai.pro).

<a id="quickstart"></a>

## ⬢ &nbsp;Quickstart — 3 steps to save 90%

```bash
# ❶ Install — one-line script (Linux / macOS / WSL2 / Android Termux)
curl -fsSL https://raw.githubusercontent.com/learningCatHD/telos-sdk/main/scripts/install.sh | bash
# …or pip:  uv pip install -U telos-sdk

# ❷ Connect — auto-detects claude-code / codex / openclaw / hermes, injects
#    config, and starts the local gateway. No changes to your agent code.
telos init

# ❸ Observe — opens an offline HTML dashboard of savings in absolute dollars
telos dashboard
```

<p align="center">
  <img src="assets/05-dashboard.png" alt="TELOS savings dashboard — absolute dollars broken down by harness / model / session" width="100%"/>
</p>

→ Detailed [installation](https://docs.telosai.pro/en/start/installation) and [quickstart](https://docs.telosai.pro/en/start/quickstart) guides, including [cc-switch coexistence](https://docs.telosai.pro/en/guides/integration-paths), live in the docs.

<a id="guarantees"></a>

## ⬢ &nbsp;Four things you actually care about

TELOS changes *what you are billed for*, not *what your agent does*.

| What you care about | The guarantee |
|---|---|
| 💰 **Token bill** | **−50% to −90% on billed input tokens.** 6-turn real session −92.3%; SWE-bench Verified −52.8% new_input / −40.5% end-to-end cost. |
| 🎯 **Agent behavior** | **Unchanged.** Same model, same prompt semantics, same outputs. SWE-bench A/B: McNemar p = 0.66, no resolved-rate regression. |
| ⚡ **Inference speed** | **Not slower — faster.** Cache hits skip re-prefilling submitted bytes, so time-to-first-token falls as the session grows. |
| 🔒 **Your data** | **Captures no content.** Gateway runs on `127.0.0.1`; the usage log records token counts only — never prompt/response text. No cloud, no telemetry. |

→ Why each guarantee holds, in detail: [**docs.telosai.pro**](https://docs.telosai.pro).

## ⬢ &nbsp;Learn more

| Topic | Where |
|---|---|
| **The protocol** — three-color bands (PIN/FOLD/DROP) and monotonic append | [concepts/protocol](https://docs.telosai.pro/en/concepts/protocol) · [concepts/bands](https://docs.telosai.pro/en/concepts/bands) |
| **Support matrix** — harnesses, frontier models, inference frameworks, cc-switch | [reference/support-matrix](https://docs.telosai.pro/en/reference/support-matrix) |
| **SWE-bench Verified A/B** — pre-registered design, statistics, full report | [benchmark/swebench](https://docs.telosai.pro/en/benchmark/swebench) |
| **Architecture & integration paths** | [concepts/architecture](https://docs.telosai.pro/en/concepts/architecture) · [guides/integration-paths](https://docs.telosai.pro/en/guides/integration-paths) |
| **CLI reference & changelog** | [reference/cli](https://docs.telosai.pro/en/reference/cli) · [CHANGELOG.md](CHANGELOG.md) |

**TELOS is open source. Run it on your own workflow — see whether that 92% is real, or just another "X× tokens" claim.**

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

<sub>📖 Full documentation at <a href="https://docs.telosai.pro">docs.telosai.pro</a></sub>
</div>
